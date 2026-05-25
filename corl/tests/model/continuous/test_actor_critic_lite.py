"""
Unit tests for corl.model.continuous.actor_critic_lite.ModelActorCriticLite.

All TFLite interpreter calls are mocked — no real .tflite files are used.

The continuous lite model supports two actor architectures:
- SAC: actor output shape is (batch, action_size * 2); predict returns tanh(mean).
- TD3: actor output shape is (batch, action_size) with tanh applied in the model;
  predict returns the raw output directly.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

# Mock ai_edge_litert before any corl imports that pull it in
_litert_mock = MagicMock()
sys.modules.setdefault("ai_edge_litert", _litert_mock)
sys.modules.setdefault("ai_edge_litert.interpreter", _litert_mock.interpreter)

from corl.model.continuous.actor_critic_lite import ModelActorCriticLite  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_interpreter(
    output: NDArray[np.float32],
    input_index: int = 0,
    output_index: int = 1,
) -> MagicMock:
    """Return a mock LiteRT interpreter with a fixed output tensor."""
    interp = MagicMock()
    interp.get_input_details.return_value = [{"index": input_index}]
    interp.get_output_details.return_value = [{"index": output_index}]
    interp.get_tensor.return_value = output
    return interp


def make_loaded_model(
    output: NDArray[np.float32],
    action_size: int = 2,
) -> ModelActorCriticLite:
    """Return a ModelActorCriticLite with a pre-loaded mock interpreter."""
    m = ModelActorCriticLite(action_size=action_size)
    m._actor_interpreter = make_interpreter(output)
    m._actor_input_index = 0
    m._actor_output_index = 1
    return m


def sac_output(action_size: int = 2) -> NDArray[np.float32]:
    """SAC actor output: [mean..., log_std...] with shape (1, action_size * 2)."""
    return np.zeros((1, action_size * 2), dtype=np.float32)


def td3_output(action_size: int = 2) -> NDArray[np.float32]:
    """TD3 actor output: tanh actions with shape (1, action_size)."""
    return np.full((1, action_size), 0.5, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> ModelActorCriticLite:
    return ModelActorCriticLite(action_size=2)


@pytest.fixture
def sac_model() -> ModelActorCriticLite:
    return make_loaded_model(output=sac_output(action_size=2), action_size=2)


@pytest.fixture
def td3_model() -> ModelActorCriticLite:
    return make_loaded_model(output=td3_output(action_size=2), action_size=2)


@pytest.fixture
def obs() -> NDArray[np.float32]:
    return np.zeros((1, 8), dtype=np.float32)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_instantiates_with_action_size(self):
        m = ModelActorCriticLite(action_size=2)
        assert isinstance(m, ModelActorCriticLite)

    def test_action_size_stored(self):
        m = ModelActorCriticLite(action_size=3)
        assert m.action_size == 3

    def test_action_size_defaults_to_zero(self):
        m = ModelActorCriticLite()
        assert m.action_size == 0

    def test_interpreter_is_none_by_default(self, model):
        assert model._actor_interpreter is None

    def test_input_index_is_none_by_default(self, model):
        assert model._actor_input_index is None

    def test_output_index_is_none_by_default(self, model):
        assert model._actor_output_index is None

    def test_model_dir_triggers_load(self):
        with patch.object(ModelActorCriticLite, "load") as mock_load:
            ModelActorCriticLite(model_dir="/some/dir")
        mock_load.assert_called_once_with("/some/dir")


# ---------------------------------------------------------------------------
# _load_interpreter()
# ---------------------------------------------------------------------------


class TestLoadInterpreter:
    def test_creates_interpreter_with_string_path(self):
        interp = make_interpreter(sac_output())
        with patch(
            "corl.model.continuous.actor_critic_lite.Interpreter",
            return_value=interp,
        ) as mock_cls:
            ModelActorCriticLite._load_interpreter(Path("/some/dir/actor.tflite"))
        mock_cls.assert_called_once_with(model_path=str(Path("/some/dir/actor.tflite")))

    def test_allocates_tensors(self):
        interp = make_interpreter(sac_output())
        with patch(
            "corl.model.continuous.actor_critic_lite.Interpreter",
            return_value=interp,
        ):
            ModelActorCriticLite._load_interpreter(Path("/f.tflite"))
        interp.allocate_tensors.assert_called_once()

    def test_returns_interpreter_and_indices(self):
        interp = make_interpreter(sac_output(), input_index=3, output_index=7)
        with patch(
            "corl.model.continuous.actor_critic_lite.Interpreter",
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
            with caplog.at_level(
                logging.INFO, logger="corl.model.continuous.actor_critic_lite"
            ):
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
# predict() — common
# ---------------------------------------------------------------------------


class TestPredictCommon:
    def test_raises_when_interpreter_not_loaded(self, model, obs):
        with pytest.raises(RuntimeError, match="load_weights"):
            model.predict(obs)

    def test_sets_input_tensor(self, sac_model, obs):
        sac_model.predict(obs)
        sac_model._actor_interpreter.set_tensor.assert_called_once_with(
            sac_model._actor_input_index, obs
        )

    def test_invokes_interpreter(self, sac_model, obs):
        sac_model.predict(obs)
        sac_model._actor_interpreter.invoke.assert_called_once()

    def test_reads_output_tensor(self, sac_model, obs):
        sac_model.predict(obs)
        sac_model._actor_interpreter.get_tensor.assert_called_once_with(
            sac_model._actor_output_index
        )

    def test_returns_float32(self, sac_model, obs):
        result = sac_model.predict(obs)
        assert result.dtype == np.float32

    def test_output_is_1d(self, sac_model, obs):
        result = sac_model.predict(obs)
        assert result.ndim == 1

    def test_output_length_equals_action_size(self, sac_model, obs):
        result = sac_model.predict(obs)
        assert len(result) == sac_model.action_size


# ---------------------------------------------------------------------------
# predict() — SAC branch (output_size == action_size * 2)
# ---------------------------------------------------------------------------


class TestPredictSAC:
    def test_result_in_minus_one_to_one(self, obs):
        """tanh(mean) must be in (-1, 1)."""
        output = np.array([[2.0, -3.0, 0.5, -0.5]], dtype=np.float32)  # 4 = 2*2
        m = make_loaded_model(output=output, action_size=2)
        result = m.predict(obs)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_uses_only_mean_half(self, obs):
        """Only the first action_size values (mean) should influence the output."""
        mean = np.array([1.5, -1.5], dtype=np.float32)
        log_std = np.array([999.0, -999.0], dtype=np.float32)
        output = np.array([np.concatenate([mean, log_std])], dtype=np.float32)
        m = make_loaded_model(output=output, action_size=2)
        result = m.predict(obs)
        np.testing.assert_allclose(result, np.tanh(mean), atol=1e-6)

    def test_zero_mean_gives_zero_action(self, obs):
        output = np.zeros((1, 4), dtype=np.float32)
        m = make_loaded_model(output=output, action_size=2)
        result = m.predict(obs)
        np.testing.assert_array_equal(result, np.zeros(2, dtype=np.float32))

    def test_large_positive_mean_saturates_to_one(self, obs):
        output = np.array([[100.0, 100.0, 0.0, 0.0]], dtype=np.float32)
        m = make_loaded_model(output=output, action_size=2)
        result = m.predict(obs)
        np.testing.assert_allclose(result, np.ones(2, dtype=np.float32), atol=1e-5)

    def test_large_negative_mean_saturates_to_minus_one(self, obs):
        output = np.array([[-100.0, -100.0, 0.0, 0.0]], dtype=np.float32)
        m = make_loaded_model(output=output, action_size=2)
        result = m.predict(obs)
        np.testing.assert_allclose(result, -np.ones(2, dtype=np.float32), atol=1e-5)

    def test_works_with_action_size_3(self, obs):
        output = np.array([[0.5, -0.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        m = make_loaded_model(output=output, action_size=3)
        result = m.predict(obs)
        assert len(result) == 3
        np.testing.assert_allclose(
            result, np.tanh([0.5, -0.5, 1.0]).astype(np.float32), atol=1e-6
        )


# ---------------------------------------------------------------------------
# predict() — TD3 branch (output_size == action_size)
# ---------------------------------------------------------------------------


class TestPredictTD3:
    def test_returns_raw_output_unchanged(self, obs):
        """TD3 actor already applies tanh; output must pass through unmodified."""
        raw = np.array([[0.3, -0.7]], dtype=np.float32)
        m = make_loaded_model(output=raw, action_size=2)
        result = m.predict(obs)
        np.testing.assert_array_equal(result, raw.flatten())

    def test_result_is_float32(self, obs):
        raw = np.array([[0.1, 0.2]], dtype=np.float32)
        m = make_loaded_model(output=raw, action_size=2)
        result = m.predict(obs)
        assert result.dtype == np.float32

    def test_output_length_equals_action_size(self, obs):
        raw = np.array([[0.1, 0.2]], dtype=np.float32)
        m = make_loaded_model(output=raw, action_size=2)
        result = m.predict(obs)
        assert len(result) == 2

    def test_extreme_values_are_not_clipped(self, obs):
        """Values from a saturated tanh model (~±1) must not be altered."""
        raw = np.array([[0.9999, -0.9999]], dtype=np.float32)
        m = make_loaded_model(output=raw, action_size=2)
        result = m.predict(obs)
        np.testing.assert_array_equal(result, raw.flatten())

    def test_works_with_action_size_3(self, obs):
        raw = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
        m = make_loaded_model(output=raw, action_size=3)
        result = m.predict(obs)
        assert len(result) == 3
        np.testing.assert_array_equal(result, raw.flatten())


# ---------------------------------------------------------------------------
# load_metadata()
# ---------------------------------------------------------------------------


class TestLoadMetadata:
    def test_sets_action_size_from_json(self, model):
        with patch("builtins.open", MagicMock()):
            with patch(
                "corl.model.continuous.actor_critic_lite.json.load",
                return_value={"action_size": 4},
            ):
                model.load_metadata("/some/dir")
        assert model.action_size == 4

    def test_reads_from_correct_path(self, model):
        m = MagicMock()
        m.__enter__ = MagicMock(return_value=MagicMock())
        m.__exit__ = MagicMock(return_value=False)
        with patch("builtins.open", return_value=m) as mock_open:
            with patch(
                "corl.model.continuous.actor_critic_lite.json.load",
                return_value={"action_size": 2},
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
                "corl.model.continuous.actor_critic_lite.json.load",
                return_value={"action_size": 2},
            ):
                with caplog.at_level(
                    logging.INFO, logger="corl.model.continuous.actor_critic_lite"
                ):
                    model.load_metadata("/some/dir")

        assert any("action_size" in r.message for r in caplog.records)
