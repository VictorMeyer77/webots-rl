"""
Unit tests for corl.model.genetic.ModelGenetic

File I/O is patched throughout — no real files are written or read.
"""

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from corl.model.genetic import ModelGenetic

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_actions(n_states: int = 4, action_dim: int = 2) -> NDArray[np.float32]:
    """Return a deterministic action table for use in tests."""
    return np.arange(n_states * action_dim, dtype=np.float32).reshape(
        n_states, action_dim
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> ModelGenetic:
    return ModelGenetic()


@pytest.fixture
def loaded_model() -> ModelGenetic:
    m = ModelGenetic()
    m.actions = make_actions()
    return m


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_instantiates(self, model):
        assert isinstance(model, ModelGenetic)

    def test_actions_is_none_by_default(self, model):
        assert model.actions is None

    def test_checkpoint_index_starts_at_zero(self, model):
        assert model.checkpoint_index == 0

    def test_each_instance_has_independent_actions(self):
        """Instance attributes must not be shared between instances."""
        m1 = ModelGenetic()
        m2 = ModelGenetic()
        m1.actions = make_actions()
        assert m2.actions is None


# ---------------------------------------------------------------------------
# load_weights()
# ---------------------------------------------------------------------------


class TestLoadWeights:
    def test_loads_array_from_npy(self, model):
        actions = make_actions()
        with patch("corl.model.genetic.np.load", return_value=actions) as mock_load:
            model.load_weights("/some/dir")

        mock_load.assert_called_once_with(
            Path("/some/dir") / "model.npy", allow_pickle=False
        )
        np.testing.assert_array_equal(model.actions, actions)

    def test_uses_correct_path(self, model):
        with patch(
            "corl.model.genetic.np.load", return_value=make_actions()
        ) as mock_load:
            model.load_weights("/my/model/dir")

        mock_load.assert_called_once_with(
            Path("/my/model/dir") / "model.npy", allow_pickle=False
        )

    def test_logs_info_on_success(self, model, caplog):
        with patch("corl.model.genetic.np.load", return_value=make_actions()):
            with caplog.at_level(logging.INFO, logger="corl.model.genetic"):
                model.load_weights("/some/dir")

        assert any("Model loaded from" in r.message for r in caplog.records)

    def test_raises_file_not_found_when_missing(self, model):
        with patch(
            "corl.model.genetic.np.load", side_effect=FileNotFoundError("no file")
        ):
            with pytest.raises(FileNotFoundError):
                model.load_weights("/missing/dir")


# ---------------------------------------------------------------------------
# save_weights()
# ---------------------------------------------------------------------------


class TestSaveWeights:
    def test_saves_to_model_npy_by_default(self, loaded_model):
        with patch("corl.model.genetic.np.save") as mock_save:
            loaded_model.save_weights("/out/dir")

        mock_save.assert_called_once_with(
            Path("/out/dir") / "model.npy", loaded_model.actions
        )

    def test_checkpoint_uses_index_in_filename(self, loaded_model):
        with patch("corl.model.genetic.np.save") as mock_save:
            loaded_model.save_weights("/out/dir", checkpoint=True)

        mock_save.assert_called_once_with(
            Path("/out/dir") / "model_ckt_0.npy", loaded_model.actions
        )

    def test_checkpoint_increments_index(self, loaded_model):
        with patch("corl.model.genetic.np.save"):
            loaded_model.save_weights("/out/dir", checkpoint=True)
            loaded_model.save_weights("/out/dir", checkpoint=True)

        assert loaded_model.checkpoint_index == 2

    def test_successive_checkpoints_use_unique_filenames(self, loaded_model):
        saved_paths = []
        with patch(
            "corl.model.genetic.np.save", side_effect=lambda p, _: saved_paths.append(p)
        ):
            loaded_model.save_weights("/out/dir", checkpoint=True)
            loaded_model.save_weights("/out/dir", checkpoint=True)
            loaded_model.save_weights("/out/dir", checkpoint=True)

        assert len(set(saved_paths)) == 3

    def test_non_checkpoint_does_not_increment_index(self, loaded_model):
        with patch("corl.model.genetic.np.save"):
            loaded_model.save_weights("/out/dir")
            loaded_model.save_weights("/out/dir")

        assert loaded_model.checkpoint_index == 0

    def test_raises_runtime_error_when_actions_none(self, model):
        with pytest.raises(RuntimeError, match="Cannot save"):
            model.save_weights("/out/dir")

    def test_logs_info_on_checkpoint(self, loaded_model, caplog):
        with patch("corl.model.genetic.np.save"):
            with caplog.at_level(logging.INFO, logger="corl.model.genetic"):
                loaded_model.save_weights("/out/dir", checkpoint=True)

        assert any("Model saved" in r.message for r in caplog.records)

    def test_logs_info_on_final_save(self, loaded_model, caplog):
        with patch("corl.model.genetic.np.save"):
            with caplog.at_level(logging.INFO, logger="corl.model.genetic"):
                loaded_model.save_weights("/out/dir")

        assert any("Model saved" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPredict:
    def test_returns_correct_action_row(self, loaded_model):
        obs = np.array([2.0, 0.0], dtype=np.float32)
        result = loaded_model.predict(obs)
        expected = make_actions()[2].astype(np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_returns_ndarray_of_int32(self, loaded_model):
        obs = np.array([0.0], dtype=np.float32)
        result = loaded_model.predict(obs)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32

    def test_first_observation_element_used_as_index(self, loaded_model):
        """The second element of the observation must be ignored."""
        obs_a = np.array([1.0, 99.0], dtype=np.float32)
        obs_b = np.array([1.0, -99.0], dtype=np.float32)
        np.testing.assert_array_equal(
            loaded_model.predict(obs_a), loaded_model.predict(obs_b)
        )

    def test_index_zero(self, loaded_model):
        obs = np.array([0.0], dtype=np.float32)
        expected = make_actions()[0].astype(np.int32)
        np.testing.assert_array_equal(loaded_model.predict(obs), expected)

    def test_last_valid_index(self, loaded_model):
        last_idx = make_actions().shape[0] - 1
        obs = np.array([float(last_idx)], dtype=np.float32)
        expected = make_actions()[last_idx].astype(np.int32)
        np.testing.assert_array_equal(loaded_model.predict(obs), expected)

    def test_raises_runtime_error_when_actions_none(self, model):
        obs = np.array([0.0], dtype=np.float32)
        with pytest.raises(RuntimeError, match="Cannot predict"):
            model.predict(obs)

    def test_raises_index_error_for_out_of_bounds(self, loaded_model):
        obs = np.array([999.0], dtype=np.float32)
        with pytest.raises(IndexError):
            loaded_model.predict(obs)

    def test_logs_debug_on_prediction(self, loaded_model, caplog):
        obs = np.array([0.0], dtype=np.float32)
        with caplog.at_level(logging.DEBUG, logger="corl.model.genetic"):
            loaded_model.predict(obs)

        assert any("Predicted action" in r.message for r in caplog.records)
