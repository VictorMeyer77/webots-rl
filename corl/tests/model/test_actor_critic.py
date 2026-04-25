"""
Unit tests for corl.model.actor_critic.ModelActorCritic.

TensorFlow is mocked at import time (pyarrow crash on macOS M2).
All Keras I/O calls are mocked — no real models are loaded, saved, or run.
"""

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

from corl.model.actor_critic import ModelActorCritic  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_keras_model(
    output: NDArray[np.float32] | None = None,
    n_outputs: int = 4,
    batch_size: int = 2,
) -> MagicMock:
    """Return a mock tf.keras.Model with configurable call output."""
    if output is None:
        output = np.zeros((batch_size, n_outputs), dtype=np.float32)
    mock = MagicMock()
    mock.return_value = MagicMock()
    mock.return_value.numpy.return_value = output
    mock.trainable_variables = [MagicMock()]
    return mock


def make_model(
    action_size: int = 4,
    actor_output: NDArray[np.float32] | None = None,
    critic_output: NDArray[np.float32] | None = None,
) -> ModelActorCritic:
    """Return a ModelActorCritic with mock actor and critic."""
    actor = make_keras_model(output=actor_output, n_outputs=action_size)
    critic = make_keras_model(
        output=critic_output
        if critic_output is not None
        else np.zeros((2, 1), dtype=np.float32),
        n_outputs=1,
    )
    return ModelActorCritic(actor=actor, critic=critic, action_size=action_size)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> ModelActorCritic:
    return make_model()


@pytest.fixture
def obs_batch() -> NDArray[np.float32]:
    return np.random.rand(2, 8).astype(np.float32)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_instantiates_with_actor_critic_action_size(self):
        m = make_model()
        assert isinstance(m, ModelActorCritic)

    def test_checkpoint_index_starts_at_zero(self, model):
        assert model.checkpoint_index == 0

    def test_action_size_stored(self):
        m = make_model(action_size=6)
        assert m.action_size == 6

    def test_actor_and_critic_stored(self):
        actor = make_keras_model()
        critic = make_keras_model(n_outputs=1)
        m = ModelActorCritic(actor=actor, critic=critic, action_size=4)
        assert m.actor is actor
        assert m.critic is critic

    def test_raises_without_any_args(self):
        with pytest.raises(ValueError, match="actor, critic, and action_size"):
            ModelActorCritic()

    def test_raises_when_actor_missing(self):
        with pytest.raises(ValueError):
            ModelActorCritic(critic=make_keras_model(), action_size=4)

    def test_raises_when_critic_missing(self):
        with pytest.raises(ValueError):
            ModelActorCritic(actor=make_keras_model(), action_size=4)

    def test_raises_when_action_size_missing(self):
        with pytest.raises(ValueError):
            ModelActorCritic(actor=make_keras_model(), critic=make_keras_model())

    def test_model_dir_triggers_load(self):
        with patch.object(ModelActorCritic, "load") as mock_load:
            ModelActorCritic(model_dir="/some/dir")
        mock_load.assert_called_once_with("/some/dir")

    def test_model_dir_warns_when_others_provided(self, caplog):
        actor = make_keras_model()
        critic = make_keras_model()
        with patch.object(ModelActorCritic, "load"):
            with caplog.at_level(logging.WARNING, logger="corl.model.actor_critic"):
                ModelActorCritic(
                    model_dir="/some/dir",
                    actor=actor,
                    critic=critic,
                    action_size=4,
                )
        assert any("ignored" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPredict:
    def test_returns_ndarray_of_int32(self, model, obs_batch):
        # softmax mock returns numpy probs
        probs = np.array(
            [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]],
            dtype=np.float32,
        )
        softmax_result = MagicMock()
        softmax_result.numpy.return_value = probs
        _tf_mock.nn.softmax.return_value = softmax_result

        result = model.predict(obs_batch)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32

    def test_output_shape_matches_batch(self, model, obs_batch):
        probs = np.full((2, 4), 0.25, dtype=np.float32)
        softmax_result = MagicMock()
        softmax_result.numpy.return_value = probs
        _tf_mock.nn.softmax.return_value = softmax_result

        result = model.predict(obs_batch)
        assert result.shape == (obs_batch.shape[0],)

    def test_actions_in_valid_range(self, model, obs_batch):
        probs = np.full((2, 4), 0.25, dtype=np.float32)
        softmax_result = MagicMock()
        softmax_result.numpy.return_value = probs
        _tf_mock.nn.softmax.return_value = softmax_result

        result = model.predict(obs_batch)
        assert all(0 <= a < model.action_size for a in result)

    def test_calls_actor_with_training_false(self, model, obs_batch):
        probs = np.full((2, 4), 0.25, dtype=np.float32)
        softmax_result = MagicMock()
        softmax_result.numpy.return_value = probs
        _tf_mock.nn.softmax.return_value = softmax_result

        model.predict(obs_batch)
        model.actor.assert_called_once_with(obs_batch, training=False)

    def test_samples_from_high_prob_action(self):
        """When one action has ~1.0 probability, it should always be chosen."""
        probs = np.array(
            [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        softmax_result = MagicMock()
        softmax_result.numpy.return_value = probs
        _tf_mock.nn.softmax.return_value = softmax_result

        m = make_model(action_size=4)
        obs = np.zeros((2, 8), dtype=np.float32)
        result = m.predict(obs)
        np.testing.assert_array_equal(result, np.array([2, 0], dtype=np.int32))


# ---------------------------------------------------------------------------
# save_weights()
# ---------------------------------------------------------------------------


class TestSaveWeights:
    def test_saves_actor_and_critic_by_default(self, model):
        model.save_weights("/out/dir")
        model.actor.save.assert_called_once_with(Path("/out/dir") / "actor.keras")
        model.critic.save.assert_called_once_with(Path("/out/dir") / "critic.keras")

    def test_checkpoint_uses_index_in_filenames(self, model):
        model.save_weights("/out/dir", checkpoint=True)
        model.actor.save.assert_called_once_with(Path("/out/dir") / "actor_ckt_0.keras")
        model.critic.save.assert_called_once_with(
            Path("/out/dir") / "critic_ckt_0.keras"
        )

    def test_checkpoint_increments_index(self, model):
        model.save_weights("/out/dir", checkpoint=True)
        model.save_weights("/out/dir", checkpoint=True)
        assert model.checkpoint_index == 2

    def test_successive_checkpoints_use_unique_filenames(self, model):
        model.save_weights("/out/dir", checkpoint=True)
        model.save_weights("/out/dir", checkpoint=True)
        actor_paths = [c.args[0] for c in model.actor.save.call_args_list]
        assert len(set(actor_paths)) == 2

    def test_non_checkpoint_does_not_increment_index(self, model):
        model.save_weights("/out/dir")
        model.save_weights("/out/dir")
        assert model.checkpoint_index == 0


# ---------------------------------------------------------------------------
# save_weights_lite()
# ---------------------------------------------------------------------------


class TestSaveWeightsLite:
    def test_writes_actor_tflite_file(self, model):
        mock_converter = MagicMock()
        mock_converter.convert.return_value = b"tflite_bytes"

        with patch(
            "corl.model.actor_critic.tf.lite.TFLiteConverter.from_keras_model",
            return_value=mock_converter,
        ):
            mock_open = MagicMock()
            with patch("builtins.open", mock_open):
                model.save_weights_lite("/out/dir")

        mock_open.assert_called_once_with(Path("/out/dir") / "actor.tflite", "wb")
        mock_open().__enter__().write.assert_called_once_with(b"tflite_bytes")

    def test_converts_actor_not_critic(self, model):
        mock_converter = MagicMock()
        mock_converter.convert.return_value = b"tflite_bytes"

        with patch(
            "corl.model.actor_critic.tf.lite.TFLiteConverter.from_keras_model",
            return_value=mock_converter,
        ) as mock_from_keras:
            with patch("builtins.open", MagicMock()):
                model.save_weights_lite("/out/dir")

        mock_from_keras.assert_called_once_with(model.actor)


# ---------------------------------------------------------------------------
# save() — also exports lite
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_calls_save_weights_lite(self, model):
        with patch.object(model, "save_metadata"):
            with patch.object(model, "save_weights_lite") as mock_lite:
                model.save("/out/dir")
        mock_lite.assert_called_once_with("/out/dir")

    def test_save_calls_save_weights_and_metadata(self, model):
        with patch.object(model, "save_weights_lite"):
            with patch.object(model, "save_metadata") as mock_meta:
                with patch.object(model, "save_weights") as mock_w:
                    from corl.model.model import Model

                    Model.save(model, "/out/dir")
        mock_w.assert_called_once()
        mock_meta.assert_called_once()


# ---------------------------------------------------------------------------
# load_weights()
# ---------------------------------------------------------------------------


class TestLoadWeights:
    def test_loads_actor_and_critic_from_correct_paths(self, model):
        actor_mock = make_keras_model()
        critic_mock = make_keras_model(n_outputs=1)

        def side_effect(path):
            if "actor" in str(path):
                return actor_mock
            return critic_mock

        with patch(
            "corl.model.actor_critic.tf.keras.models.load_model",
            side_effect=side_effect,
        ) as mock_load:
            model.load_weights("/some/dir")

        calls = [str(c.args[0]) for c in mock_load.call_args_list]
        assert str(Path("/some/dir") / "actor.keras") in calls
        assert str(Path("/some/dir") / "critic.keras") in calls

    def test_actor_and_critic_set_after_load(self, model):
        actor_mock = make_keras_model()
        critic_mock = make_keras_model(n_outputs=1)

        def side_effect(path):
            if "actor" in str(path):
                return actor_mock
            return critic_mock

        with patch(
            "corl.model.actor_critic.tf.keras.models.load_model",
            side_effect=side_effect,
        ):
            model.load_weights("/some/dir")

        assert model.actor is actor_mock
        assert model.critic is critic_mock

    def test_logs_info_on_success(self, model, caplog):
        with patch(
            "corl.model.actor_critic.tf.keras.models.load_model",
            return_value=make_keras_model(),
        ):
            with caplog.at_level(logging.INFO, logger="corl.model.actor_critic"):
                model.load_weights("/some/dir")

        assert any("Loaded" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# load_metadata()
# ---------------------------------------------------------------------------


class TestLoadMetadata:
    def test_sets_action_size_from_json(self, model):
        with patch("builtins.open", MagicMock()):
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
        with patch("builtins.open", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                model.load_metadata("/missing/dir")

    def test_logs_info_on_success(self, model, caplog):
        with patch("builtins.open", MagicMock()):
            with patch("json.load", return_value={"action_size": 4}):
                with caplog.at_level(logging.INFO, logger="corl.model.actor_critic"):
                    model.load_metadata("/some/dir")

        assert any("action_size" in r.message for r in caplog.records)
