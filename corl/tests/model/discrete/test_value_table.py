"""
Unit tests for corl.model.discrete.value_table.ModelValueTable.

File I/O is patched throughout — no real files are written or read.
"""

import json
import logging
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from corl.model.discrete.value_table import ModelValueTable

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def make_model(
    observation_cardinality: int = 3,
    observation_size: int = 2,
    action_size: int = 4,
) -> ModelValueTable:
    return ModelValueTable(
        observation_cardinality=observation_cardinality,
        observation_size=observation_size,
        action_size=action_size,
    )


@pytest.fixture
def model() -> ModelValueTable:
    return make_model()


@pytest.fixture
def obs(model) -> NDArray[np.float32]:
    """A valid observation for the default model (cardinality=3, size=2)."""
    return np.array([1.0, 2.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_instantiates_from_dimensions(self):
        m = make_model(observation_cardinality=3, observation_size=2, action_size=4)
        assert isinstance(m, ModelValueTable)

    def test_value_table_shape(self):
        m = make_model(observation_cardinality=3, observation_size=2, action_size=4)
        assert m.value_table.shape == (9, 4)  # 3**2 x 4

    def test_value_table_zero_initialised(self):
        m = make_model()
        np.testing.assert_array_equal(m.value_table, 0.0)

    def test_value_table_dtype_is_float32(self):
        m = make_model()
        assert m.value_table.dtype == np.float32

    def test_checkpoint_index_starts_at_zero(self):
        m = make_model()
        assert m.checkpoint_index == 0

    def test_dimensions_stored(self):
        m = make_model(observation_cardinality=5, observation_size=3, action_size=6)
        assert m.observation_cardinality == 5
        assert m.observation_size == 3
        assert m.action_size == 6

    def test_raises_when_no_args(self):
        with pytest.raises(ValueError, match="must be provided"):
            ModelValueTable()

    def test_raises_when_only_observation_cardinality(self):
        with pytest.raises(ValueError):
            ModelValueTable(observation_cardinality=3)

    def test_raises_when_only_action_size(self):
        with pytest.raises(ValueError):
            ModelValueTable(action_size=4)

    def test_model_dir_triggers_load(self):
        with patch.object(ModelValueTable, "load") as mock_load:
            ModelValueTable(model_dir="/some/dir")
        mock_load.assert_called_once_with("/some/dir")

    def test_model_dir_warns_when_dims_also_provided(self, caplog):
        with patch.object(ModelValueTable, "load"):
            with caplog.at_level(
                logging.WARNING, logger="corl.model.discrete.value_table"
            ):
                ModelValueTable(
                    model_dir="/some/dir",
                    observation_cardinality=3,
                    observation_size=2,
                    action_size=4,
                )
        assert any("ignored" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# load_weights()
# ---------------------------------------------------------------------------


class TestLoadWeights:
    def test_loads_array_from_npy(self, model):
        table = np.ones((9, 4), dtype=np.float32)
        with patch(
            "corl.model.discrete.value_table.np.load", return_value=table
        ) as mock_load:
            model.load_weights("/some/dir")

        mock_load.assert_called_once_with(
            Path("/some/dir") / "model.npy", allow_pickle=False
        )
        np.testing.assert_array_equal(model.value_table, table)

    def test_uses_correct_path(self, model):
        with patch(
            "corl.model.discrete.value_table.np.load", return_value=np.zeros((9, 4))
        ) as mock_load:
            model.load_weights("/my/model/dir")

        mock_load.assert_called_once_with(
            Path("/my/model/dir") / "model.npy", allow_pickle=False
        )

    def test_logs_info_on_success(self, model, caplog):
        with patch(
            "corl.model.discrete.value_table.np.load", return_value=np.zeros((9, 4))
        ):
            with caplog.at_level(
                logging.INFO, logger="corl.model.discrete.value_table"
            ):
                model.load_weights("/some/dir")

        assert any("Model loaded from" in r.message for r in caplog.records)

    def test_raises_file_not_found(self, model):
        with patch(
            "corl.model.discrete.value_table.np.load",
            side_effect=FileNotFoundError("no file"),
        ):
            with pytest.raises(FileNotFoundError):
                model.load_weights("/missing/dir")


# ---------------------------------------------------------------------------
# save_weights()
# ---------------------------------------------------------------------------


class TestSaveWeights:
    def test_saves_to_model_npy_by_default(self, model):
        with patch("corl.model.discrete.value_table.np.save") as mock_save:
            model.save_weights("/out/dir")

        mock_save.assert_called_once_with(
            Path("/out/dir") / "model.npy", model.value_table
        )

    def test_checkpoint_uses_index_in_filename(self, model):
        with patch("corl.model.discrete.value_table.np.save") as mock_save:
            model.save_weights("/out/dir", checkpoint=True)

        mock_save.assert_called_once_with(
            Path("/out/dir") / "model_ckt_0.npy", model.value_table
        )

    def test_checkpoint_increments_index(self, model):
        with patch("corl.model.discrete.value_table.np.save"):
            model.save_weights("/out/dir", checkpoint=True)
            model.save_weights("/out/dir", checkpoint=True)

        assert model.checkpoint_index == 2

    def test_successive_checkpoints_use_unique_filenames(self, model):
        saved_paths = []
        with patch(
            "corl.model.discrete.value_table.np.save",
            side_effect=lambda p, _: saved_paths.append(p),
        ):
            model.save_weights("/out/dir", checkpoint=True)
            model.save_weights("/out/dir", checkpoint=True)
            model.save_weights("/out/dir", checkpoint=True)

        assert len(set(saved_paths)) == 3

    def test_non_checkpoint_does_not_increment_index(self, model):
        with patch("corl.model.discrete.value_table.np.save"):
            model.save_weights("/out/dir")
            model.save_weights("/out/dir")

        assert model.checkpoint_index == 0


# ---------------------------------------------------------------------------
# load_metadata()
# ---------------------------------------------------------------------------


class TestLoadMetadata:
    def _patch_open(self, payload: dict):
        return patch(
            "builtins.open",
            mock_open(read_data=json.dumps(payload)),
        )

    def test_sets_all_dimensions(self, model):
        payload = {
            "observation_cardinality": 5,
            "observation_size": 3,
            "action_size": 6,
        }
        with self._patch_open(payload):
            model.load_metadata("/some/dir")

        assert model.observation_cardinality == 5
        assert model.observation_size == 3
        assert model.action_size == 6

    def test_reads_from_correct_path(self, model):
        payload = {
            "observation_cardinality": 3,
            "observation_size": 2,
            "action_size": 4,
        }
        with patch("builtins.open", mock_open(read_data=json.dumps(payload))) as m:
            model.load_metadata("/some/dir")

        m.assert_called_once_with(Path("/some/dir") / "metadata.json", "r")

    def test_raises_file_not_found(self, model):
        with patch("builtins.open", side_effect=FileNotFoundError("no file")):
            with pytest.raises(FileNotFoundError):
                model.load_metadata("/missing/dir")

    def test_raises_on_malformed_json(self, model):
        with patch("builtins.open", mock_open(read_data="not json {{{")):
            with pytest.raises(json.JSONDecodeError):
                model.load_metadata("/bad/dir")

    def test_raises_on_missing_key(self, model):
        with self._patch_open({"observation_cardinality": 3}):
            with pytest.raises(KeyError):
                model.load_metadata("/incomplete/dir")

    def test_casts_values_to_int(self, model):
        payload = {
            "observation_cardinality": "3",
            "observation_size": "2",
            "action_size": "4",
        }
        with self._patch_open(payload):
            model.load_metadata("/some/dir")

        assert isinstance(model.observation_cardinality, int)
        assert isinstance(model.observation_size, int)
        assert isinstance(model.action_size, int)


# ---------------------------------------------------------------------------
# observation_to_index()
# ---------------------------------------------------------------------------


class TestObservationToIndex:
    def test_zero_observation_maps_to_zero(self, model):
        obs = np.array([0.0, 0.0], dtype=np.float32)
        assert model.observation_to_index(obs) == 0

    def test_known_index_cardinality_3_size_2(self, model):
        # shape (3, 3): row-major index of (1, 2) = 1*3 + 2 = 5
        obs = np.array([1.0, 2.0], dtype=np.float32)
        assert model.observation_to_index(obs) == 5

    def test_last_valid_observation(self, model):
        # max index: (2, 2) → 2*3 + 2 = 8
        obs = np.array([2.0, 2.0], dtype=np.float32)
        assert model.observation_to_index(obs) == 8

    def test_float_observation_truncated_to_int(self, model):
        obs_float = np.array([1.9, 2.9], dtype=np.float32)
        obs_int = np.array([1.0, 2.0], dtype=np.float32)
        assert model.observation_to_index(obs_float) == model.observation_to_index(
            obs_int
        )

    def test_raises_on_wrong_length(self, model):
        obs = np.array([1.0], dtype=np.float32)
        with pytest.raises(ValueError, match="does not match"):
            model.observation_to_index(obs)

    def test_raises_on_negative_value(self, model):
        obs = np.array([-1.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match=r"\[0,"):
            model.observation_to_index(obs)

    def test_raises_on_value_equal_to_cardinality(self, model):
        obs = np.array([3.0, 0.0], dtype=np.float32)  # cardinality=3
        with pytest.raises(ValueError):
            model.observation_to_index(obs)

    def test_single_dimension(self):
        m = make_model(observation_cardinality=5, observation_size=1, action_size=2)
        for i in range(5):
            obs = np.array([float(i)], dtype=np.float32)
            assert m.observation_to_index(obs) == i

    def test_index_is_int(self, model, obs):
        result = model.observation_to_index(obs)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPredict:
    def test_returns_int32_array(self, model, obs):
        result = model.predict(obs)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32

    def test_returns_greedy_action(self, model):
        # Set action 2 as the best for state (0, 0)
        model.value_table[0] = np.array([0.1, 0.2, 0.9, 0.0], dtype=np.float32)
        obs = np.array([0.0, 0.0], dtype=np.float32)
        result = model.predict(obs)
        assert int(result) == 2

    def test_tie_broken_randomly(self):
        m = make_model(observation_cardinality=2, observation_size=1, action_size=3)
        m.value_table[0] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        obs = np.array([0.0], dtype=np.float32)
        results = {int(m.predict(obs)) for _ in range(200)}
        assert len(results) > 1  # tie must be broken non-deterministically

    def test_uses_correct_state_row(self, model):
        model.value_table[5] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        obs = np.array([1.0, 2.0], dtype=np.float32)  # maps to index 5
        assert int(model.predict(obs)) == 3

    def test_all_zeros_returns_valid_action(self, model, obs):
        result = int(model.predict(obs))
        assert 0 <= result < model.action_size


# ---------------------------------------------------------------------------
# epsilon_greedy_policy()
# ---------------------------------------------------------------------------


class TestEpsilonGreedyPolicy:
    def test_returns_int(self, model, obs):
        result = model.epsilon_greedy_policy(obs, epsilon=0.0)
        assert isinstance(result, int)

    def test_greedy_when_epsilon_zero(self, model):
        model.value_table[0] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        obs = np.array([0.0, 0.0], dtype=np.float32)
        results = {model.epsilon_greedy_policy(obs, epsilon=0.0) for _ in range(20)}
        assert results == {3}

    def test_random_when_epsilon_one(self, model, obs):
        results = {model.epsilon_greedy_policy(obs, epsilon=1.0) for _ in range(200)}
        assert len(results) > 1

    def test_action_in_valid_range(self, model, obs):
        for _ in range(50):
            a = model.epsilon_greedy_policy(obs, epsilon=0.5)
            assert 0 <= a < model.action_size
