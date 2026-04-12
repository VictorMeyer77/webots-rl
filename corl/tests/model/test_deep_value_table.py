"""
Unit tests for corl.model.deep_value_table.ModelDeepValueTable.

TensorFlow is mocked at import time (pyarrow crash on macOS M2).
All Keras I/O calls are mocked — no real models are loaded, saved, or run.
"""

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

# Mock TF before any corl imports that pull it in
_tf_mock = MagicMock()
sys.modules.setdefault("tensorflow", _tf_mock)
sys.modules.setdefault("tensorflow.lite", _tf_mock.lite)
sys.modules.setdefault("tensorflow.keras", _tf_mock.keras)

from corl.model.deep_value_table import ModelDeepValueTable  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_keras_model(
    output: NDArray[np.float32] | None = None,
    n_actions: int = 4,
    batch_size: int = 2,
) -> MagicMock:
    """Return a mock tf.keras.Model with configurable call output."""
    if output is None:
        output = np.zeros((batch_size, n_actions), dtype=np.float32)
    mock = MagicMock()
    mock.return_value = MagicMock()
    mock.return_value.numpy.return_value = output
    mock.fit.return_value = MagicMock(history={"loss": [0.5]})
    mock.get_weights.return_value = [np.ones((2, 2))]
    return mock


def make_model(n_actions: int = 4, batch_size: int = 2) -> ModelDeepValueTable:
    """Return a ModelDeepValueTable with a mock keras model."""
    keras_mock = make_keras_model(n_actions=n_actions, batch_size=batch_size)
    return ModelDeepValueTable(weights=keras_mock, action_size=n_actions)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> ModelDeepValueTable:
    return make_model()


@pytest.fixture
def obs_batch() -> NDArray[np.float32]:
    return np.random.rand(2, 8).astype(np.float32)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_instantiates_with_weights_and_action_size(self):
        m = make_model()
        assert isinstance(m, ModelDeepValueTable)

    def test_checkpoint_index_starts_at_zero(self, model):
        assert model.checkpoint_index == 0

    def test_action_size_stored(self):
        m = make_model(n_actions=6)
        assert m.action_size == 6

    def test_raises_without_weights_or_action_size(self):
        with pytest.raises(ValueError, match="weights and action_size"):
            ModelDeepValueTable()

    def test_raises_when_only_weights_provided(self):
        with pytest.raises(ValueError):
            ModelDeepValueTable(weights=make_keras_model())

    def test_raises_when_only_action_size_provided(self):
        with pytest.raises(ValueError):
            ModelDeepValueTable(action_size=4)

    def test_model_dir_triggers_load(self):
        with patch.object(ModelDeepValueTable, "load") as mock_load:
            ModelDeepValueTable(model_dir="/some/dir")
        mock_load.assert_called_once_with("/some/dir")

    def test_model_dir_warns_when_weights_also_provided(self, caplog):
        keras_mock = make_keras_model()
        with patch.object(ModelDeepValueTable, "load"):
            with caplog.at_level(logging.WARNING, logger="corl.model.deep_value_table"):
                ModelDeepValueTable(
                    model_dir="/some/dir", weights=keras_mock, action_size=4
                )
        assert any("ignored" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# load_weights()
# ---------------------------------------------------------------------------


class TestLoadWeights:
    def test_loads_from_correct_path(self, model):
        keras_mock = make_keras_model()
        with patch(
            "corl.model.deep_value_table.tf.keras.models.load_model",
            return_value=keras_mock,
        ) as mock_load:
            model.load_weights("/some/dir")

        mock_load.assert_called_once_with(Path("/some/dir") / "model.keras")

    def test_weights_attribute_set_after_load(self, model):
        keras_mock = make_keras_model()
        with patch(
            "corl.model.deep_value_table.tf.keras.models.load_model",
            return_value=keras_mock,
        ):
            model.load_weights("/some/dir")

        assert model.weights is keras_mock

    def test_logs_info_on_success(self, model, caplog):
        with patch(
            "corl.model.deep_value_table.tf.keras.models.load_model",
            return_value=make_keras_model(),
        ):
            with caplog.at_level(logging.INFO, logger="corl.model.deep_value_table"):
                model.load_weights("/some/dir")

        assert any("Loaded" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# save_weights()
# ---------------------------------------------------------------------------


class TestSaveWeights:
    def test_saves_to_model_keras_by_default(self, model):
        model.save_weights("/out/dir")
        model.weights.save.assert_called_once_with(Path("/out/dir") / "model.keras")

    def test_checkpoint_uses_index_in_filename(self, model):
        model.save_weights("/out/dir", checkpoint=True)
        model.weights.save.assert_called_once_with(
            Path("/out/dir") / "model_ckt_0.keras"
        )

    def test_checkpoint_increments_index(self, model):
        model.save_weights("/out/dir", checkpoint=True)
        model.save_weights("/out/dir", checkpoint=True)
        assert model.checkpoint_index == 2

    def test_successive_checkpoints_use_unique_filenames(self, model):
        model.save_weights("/out/dir", checkpoint=True)
        model.save_weights("/out/dir", checkpoint=True)
        paths = [c.args[0] for c in model.weights.save.call_args_list]
        assert len(set(paths)) == 2

    def test_non_checkpoint_does_not_increment_index(self, model):
        model.save_weights("/out/dir")
        model.save_weights("/out/dir")
        assert model.checkpoint_index == 0


# ---------------------------------------------------------------------------
# save_weights_lite()
# ---------------------------------------------------------------------------


class TestSaveWeightsLite:
    def test_writes_tflite_file(self, model):
        mock_converter = MagicMock()
        mock_converter.convert.return_value = b"tflite_bytes"

        with patch(
            "corl.model.deep_value_table.tf.lite.TFLiteConverter.from_keras_model",
            return_value=mock_converter,
        ):
            mock_open = MagicMock()
            with patch("builtins.open", mock_open):
                model.save_weights_lite("/out/dir")

        mock_open.assert_called_once_with(Path("/out/dir") / "model.tflite", "wb")
        mock_open().__enter__().write.assert_called_once_with(b"tflite_bytes")

    def test_raises_when_weights_not_initialised(self):
        m = ModelDeepValueTable.__new__(ModelDeepValueTable)
        m.checkpoint_index = 0
        with pytest.raises((ValueError, AttributeError)):
            m.save_weights_lite("/out/dir")


# ---------------------------------------------------------------------------
# save() — also exports lite
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_calls_save_weights_lite(self, model):
        with patch.object(model, "save_metadata"):
            with patch.object(model, "save_weights_lite") as mock_lite:
                model.save("/out/dir")
        mock_lite.assert_called_once_with("/out/dir")

    def test_save_calls_parent_save(self, model):
        with patch.object(model, "save_weights_lite"):
            with patch.object(model, "save_metadata") as mock_meta:
                with patch.object(model, "save_weights") as mock_w:
                    from corl.model.model import Model

                    Model.save(model, "/out/dir")
        mock_w.assert_called_once()
        mock_meta.assert_called_once()


# ---------------------------------------------------------------------------
# load_metadata()
# ---------------------------------------------------------------------------


class TestLoadMetadata:
    def test_sets_action_size_from_json(self, model):
        payload = json.dumps({"action_size": 8})
        with patch(
            "builtins.open",
            MagicMock(
                return_value=MagicMock(
                    __enter__=MagicMock(
                        return_value=MagicMock(read=MagicMock(return_value=payload))
                    ),
                    __exit__=MagicMock(return_value=False),
                )
            ),
        ):
            with patch("json.load", return_value={"action_size": 8}):
                model.load_metadata("/some/dir")

        assert model.action_size == 8

    def test_reads_from_correct_path(self, model):
        m = MagicMock()
        m.__enter__ = MagicMock(return_value=MagicMock())
        m.__exit__ = MagicMock(return_value=False)
        with patch("builtins.open", return_value=m) as mock_open:
            with patch("json.load", return_value={"action_size": 4}):
                model.load_metadata("/some/dir")

        mock_open.assert_called_once_with(Path("/some/dir") / "metadata.json", "r")

    def test_raises_file_not_found(self, model):
        with patch("builtins.open", side_effect=FileNotFoundError("no file")):
            with pytest.raises(FileNotFoundError):
                model.load_metadata("/missing/dir")


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPredict:
    def test_returns_ndarray_of_int32(self, model, obs_batch):
        result = model.predict(obs_batch)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32

    def test_returns_argmax_per_sample(self):
        q = np.array([[0.1, 0.9, 0.0, 0.0], [0.8, 0.1, 0.0, 0.1]], dtype=np.float32)
        keras_mock = make_keras_model(output=q)
        m = ModelDeepValueTable(weights=keras_mock, action_size=4)
        obs = np.zeros((2, 8), dtype=np.float32)
        result = m.predict(obs)
        np.testing.assert_array_equal(result, np.array([1, 0], dtype=np.int32))

    def test_output_shape_matches_batch(self, model, obs_batch):
        result = model.predict(obs_batch)
        assert result.shape == (obs_batch.shape[0],)

    def test_raises_when_weights_not_initialised(self):
        m = ModelDeepValueTable.__new__(ModelDeepValueTable)
        m.checkpoint_index = 0
        obs = np.zeros((1, 4), dtype=np.float32)
        with pytest.raises((ValueError, AttributeError)):
            m.predict(obs)

    def test_calls_model_with_training_false(self, model, obs_batch):
        model.predict(obs_batch)
        model.weights.assert_called_once_with(obs_batch, training=False)


# ---------------------------------------------------------------------------
# epsilon_greedy_policy()
# ---------------------------------------------------------------------------


class TestEpsilonGreedyPolicy:
    def test_returns_ndarray_of_int32(self, model, obs_batch):
        result = model.epsilon_greedy_policy(obs_batch, epsilon=0.0)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32

    def test_output_shape_matches_batch(self, model, obs_batch):
        result = model.epsilon_greedy_policy(obs_batch, epsilon=0.0)
        assert result.shape == (obs_batch.shape[0],)

    def test_greedy_when_epsilon_zero(self):
        q = np.array([[0.1, 0.9, 0.0, 0.0], [0.8, 0.1, 0.0, 0.1]], dtype=np.float32)
        keras_mock = make_keras_model(output=q)
        m = ModelDeepValueTable(weights=keras_mock, action_size=4)
        obs = np.zeros((2, 8), dtype=np.float32)
        result = m.epsilon_greedy_policy(obs, epsilon=0.0)
        np.testing.assert_array_equal(result, np.array([1, 0], dtype=np.int32))

    def test_all_random_when_epsilon_one(self, model, obs_batch):
        np.random.seed(0)
        result = model.epsilon_greedy_policy(obs_batch, epsilon=1.0)
        assert result.shape == (obs_batch.shape[0],)
        assert all(0 <= a < model.action_size for a in result)

    def test_actions_in_valid_range(self, model, obs_batch):
        result = model.epsilon_greedy_policy(obs_batch, epsilon=0.5)
        assert all(0 <= a < model.action_size for a in result)
