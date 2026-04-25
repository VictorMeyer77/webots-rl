"""
Unit tests for corl.model.actor_critic_lite.ModelActorCriticLite.

TensorFlow is mocked at import time (pyarrow crash on macOS M2).
All TFLite interpreter calls are mocked — no real .tflite files are used.
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

from corl.model.actor_critic_lite import ModelActorCriticLite  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_interpreter(
    logits: NDArray[np.float32] | None = None,
    input_index: int = 0,
    output_index: int = 1,
) -> MagicMock:
    """Return a mock TFLite interpreter with configurable output."""
    if logits is None:
        logits = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    interp = MagicMock()
    interp.get_input_details.return_value = [{"index": input_index}]
    interp.get_output_details.return_value = [{"index": output_index}]
    interp.get_tensor.return_value = logits
    return interp


def make_loaded_model(
    logits: NDArray[np.float32] | None = None,
    action_size: int = 4,
) -> ModelActorCriticLite:
    """Return a ModelActorCriticLite with a pre-loaded mock interpreter."""
    m = ModelActorCriticLite(action_size=action_size)
    m._actor_interpreter = make_interpreter(logits)
    m._actor_input_index = 0
    m._actor_output_index = 1
    return m


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> ModelActorCriticLite:
    return ModelActorCriticLite(action_size=4)


@pytest.fixture
def loaded_model() -> ModelActorCriticLite:
    return make_loaded_model()


@pytest.fixture
def obs() -> NDArray[np.float32]:
    return np.zeros((1, 8), dtype=np.float32)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_instantiates_with_action_size(self):
        m = ModelActorCriticLite(action_size=4)
        assert isinstance(m, ModelActorCriticLite)

    def test_action_size_stored(self):
        m = ModelActorCriticLite(action_size=6)
        assert m.action_size == 6

    def test_action_size_defaults_to_zero(self):
        m = ModelActorCriticLite()
        assert m.action_size == 0

    def test_interpreter_is_none_by_default(self, model):
        assert model._actor_interpreter is None

    def test_input_index_is_none_by_default(self, model):
        assert model._actor_input_index is None

    def test_output_index_is_none_by_default(self, model):
        assert model._actor_output_index is None

    def test_checkpoint_index_starts_at_zero(self, model):
        assert model.checkpoint_index == 0

    def test_model_dir_triggers_load(self):
        with patch.object(ModelActorCriticLite, "load") as mock_load:
            ModelActorCriticLite(model_dir="/some/dir")
        mock_load.assert_called_once_with("/some/dir")


# ---------------------------------------------------------------------------
# _load_interpreter()
# ---------------------------------------------------------------------------


class TestLoadInterpreter:
    def test_creates_interpreter_with_string_path(self):
        interp = make_interpreter()
        with patch(
            "corl.model.actor_critic_lite.tf.lite.Interpreter",
            return_value=interp,
        ) as mock_cls:
            ModelActorCriticLite._load_interpreter(Path("/some/dir/actor.tflite"))
        mock_cls.assert_called_once_with(model_path=str(Path("/some/dir/actor.tflite")))

    def test_allocates_tensors(self):
        interp = make_interpreter()
        with patch(
            "corl.model.actor_critic_lite.tf.lite.Interpreter",
            return_value=interp,
        ):
            ModelActorCriticLite._load_interpreter(Path("/f.tflite"))
        interp.allocate_tensors.assert_called_once()

    def test_returns_interpreter_and_indices(self):
        interp = make_interpreter(input_index=3, output_index=7)
        with patch(
            "corl.model.actor_critic_lite.tf.lite.Interpreter",
            return_value=interp,
        ):
            result = ModelActorCriticLite._load_interpreter(Path("/f.tflite"))
        assert result == (interp, 3, 7)


# ---------------------------------------------------------------------------
# load_weights()
# ---------------------------------------------------------------------------


class TestLoadWeights:
    def test_loads_actor_tflite_from_correct_path(self, model):
        with patch.object(
            ModelActorCriticLite,
            "_load_interpreter",
            return_value=(MagicMock(), 0, 1),
        ) as mock_load:
            model.load_weights("/some/dir")

        mock_load.assert_called_once_with(Path("/some/dir") / "actor.tflite")

    def test_stores_interpreter_and_indices(self, model):
        interp = MagicMock()
        with patch.object(
            ModelActorCriticLite,
            "_load_interpreter",
            return_value=(interp, 5, 9),
        ):
            model.load_weights("/some/dir")

        assert model._actor_interpreter is interp
        assert model._actor_input_index == 5
        assert model._actor_output_index == 9

    def test_logs_info_on_success(self, model, caplog):
        with patch.object(
            ModelActorCriticLite,
            "_load_interpreter",
            return_value=(MagicMock(), 0, 1),
        ):
            with caplog.at_level(logging.INFO, logger="corl.model.actor_critic_lite"):
                model.load_weights("/some/dir")

        assert any("Loaded" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# save_weights()
# ---------------------------------------------------------------------------


class TestSaveWeights:
    def test_always_raises_runtime_error(self, model):
        with pytest.raises(RuntimeError, match="ModelActorCritic"):
            model.save_weights("/out/dir")

    def test_raises_with_checkpoint_true(self, model):
        with pytest.raises(RuntimeError):
            model.save_weights("/out/dir", checkpoint=True)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPredict:
    def test_raises_when_interpreter_not_loaded(self, model, obs):
        with pytest.raises(RuntimeError, match="load_weights"):
            model.predict(obs)

    def test_returns_int32_scalar_array(self, loaded_model, obs):
        result = loaded_model.predict(obs)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32

    def test_returns_scalar(self, loaded_model, obs):
        result = loaded_model.predict(obs)
        assert result.ndim == 0

    def test_action_in_valid_range(self, loaded_model, obs):
        result = loaded_model.predict(obs)
        assert 0 <= int(result) < loaded_model.action_size

    def test_sets_input_tensor(self, loaded_model, obs):
        loaded_model.predict(obs)
        loaded_model._actor_interpreter.set_tensor.assert_called_once_with(
            loaded_model._actor_input_index, obs
        )

    def test_invokes_interpreter(self, loaded_model, obs):
        loaded_model.predict(obs)
        loaded_model._actor_interpreter.invoke.assert_called_once()

    def test_reads_output_tensor(self, loaded_model, obs):
        loaded_model.predict(obs)
        loaded_model._actor_interpreter.get_tensor.assert_called_once_with(
            loaded_model._actor_output_index
        )

    def test_samples_deterministically_from_peaked_logits(self):
        """When one logit is very large, that action should always win."""
        logits = np.array([[0.0, 0.0, 100.0, 0.0]], dtype=np.float32)
        m = make_loaded_model(logits=logits, action_size=4)
        obs = np.zeros((1, 8), dtype=np.float32)
        result = m.predict(obs)
        assert int(result) == 2

    def test_softmax_is_numerically_stable(self):
        """Large logits should not cause overflow in exp()."""
        logits = np.array([[1000.0, 1000.0, 1001.0, 1000.0]], dtype=np.float32)
        m = make_loaded_model(logits=logits, action_size=4)
        obs = np.zeros((1, 8), dtype=np.float32)
        result = m.predict(obs)
        assert 0 <= int(result) < 4

    def test_uniform_logits_produce_valid_action(self):
        """Equal logits should still produce a valid action."""
        logits = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        m = make_loaded_model(logits=logits, action_size=4)
        obs = np.zeros((1, 8), dtype=np.float32)
        result = m.predict(obs)
        assert 0 <= int(result) < 4


# ---------------------------------------------------------------------------
# load_metadata()
# ---------------------------------------------------------------------------


class TestLoadMetadata:
    def test_sets_action_size_from_json(self, model):
        with patch("builtins.open", MagicMock()):
            with patch(
                "corl.model.actor_critic_lite.json.load",
                return_value={"action_size": 8},
            ):
                model.load_metadata("/some/dir")
        assert model.action_size == 8

    def test_reads_from_correct_path(self, model):
        m = MagicMock()
        m.__enter__ = MagicMock(return_value=MagicMock())
        m.__exit__ = MagicMock(return_value=False)
        with patch("builtins.open", return_value=m) as mock_open:
            with patch(
                "corl.model.actor_critic_lite.json.load",
                return_value={"action_size": 4},
            ):
                model.load_metadata("/some/dir")

        mock_open.assert_called_once_with(Path("/some/dir") / "metadata.json", "r")

    def test_raises_file_not_found(self, model):
        with patch("builtins.open", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                model.load_metadata("/missing/dir")

    def test_logs_info_on_success(self, model, caplog):
        with patch("builtins.open", MagicMock()):
            with patch(
                "corl.model.actor_critic_lite.json.load",
                return_value={"action_size": 4},
            ):
                with caplog.at_level(
                    logging.INFO, logger="corl.model.actor_critic_lite"
                ):
                    model.load_metadata("/some/dir")

        assert any("action_size" in r.message for r in caplog.records)
