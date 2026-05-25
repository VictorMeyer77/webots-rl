"""
Unit tests for corl.model.discrete.deep_value_table_lite.ModelDeepValueTableLite.

All TFLite interpreter calls are mocked — no real .tflite files are used.
"""

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

from corl.model.discrete.deep_value_table_lite import (  # noqa: E402
    ModelDeepValueTableLite,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_interpreter(q_values: NDArray[np.float32] | None = None) -> MagicMock:
    """Return a mock TFLite interpreter with configurable output."""
    if q_values is None:
        q_values = np.array([[0.1, 0.9, 0.3, 0.0]], dtype=np.float32)
    interp = MagicMock()
    interp.get_input_details.return_value = [{"index": 0}]
    interp.get_output_details.return_value = [{"index": 1}]
    interp.get_tensor.return_value = q_values
    return interp


def make_loaded_model(
    q_values: NDArray[np.float32] | None = None,
) -> ModelDeepValueTableLite:
    """Return a ModelDeepValueTableLite with a pre-loaded mock interpreter."""
    m = ModelDeepValueTableLite()
    m._tflite_interpreter = make_interpreter(q_values)
    m._tflite_input_index = 0
    m._tflite_output_index = 1
    return m


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> ModelDeepValueTableLite:
    return ModelDeepValueTableLite()


@pytest.fixture
def loaded_model() -> ModelDeepValueTableLite:
    return make_loaded_model()


@pytest.fixture
def obs() -> NDArray[np.float32]:
    return np.zeros((1, 8), dtype=np.float32)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_instantiates(self, model):
        assert isinstance(model, ModelDeepValueTableLite)

    def test_interpreter_is_none_by_default(self, model):
        assert model._tflite_interpreter is None

    def test_input_index_is_none_by_default(self, model):
        assert model._tflite_input_index is None

    def test_output_index_is_none_by_default(self, model):
        assert model._tflite_output_index is None


# ---------------------------------------------------------------------------
# load_weights()
# ---------------------------------------------------------------------------


class TestLoadWeights:
    def _make_interp_mock(self) -> MagicMock:
        interp = make_interpreter()
        return interp

    def test_creates_interpreter_with_correct_path(self, model):
        interp = self._make_interp_mock()
        with patch(
            "corl.model.discrete.deep_value_table_lite.Interpreter",
            return_value=interp,
        ) as mock_cls:
            model.load_weights("/some/dir")

        mock_cls.assert_called_once_with(
            model_path=str(Path("/some/dir") / "model.tflite")
        )

    def test_allocates_tensors(self, model):
        interp = self._make_interp_mock()
        with patch(
            "corl.model.discrete.deep_value_table_lite.Interpreter",
            return_value=interp,
        ):
            model.load_weights("/some/dir")

        interp.allocate_tensors.assert_called_once()

    def test_stores_interpreter(self, model):
        interp = self._make_interp_mock()
        with patch(
            "corl.model.discrete.deep_value_table_lite.Interpreter",
            return_value=interp,
        ):
            model.load_weights("/some/dir")

        assert model._tflite_interpreter is interp

    def test_stores_input_index(self, model):
        interp = self._make_interp_mock()
        with patch(
            "corl.model.discrete.deep_value_table_lite.Interpreter",
            return_value=interp,
        ):
            model.load_weights("/some/dir")

        assert model._tflite_input_index == 0

    def test_stores_output_index(self, model):
        interp = self._make_interp_mock()
        with patch(
            "corl.model.discrete.deep_value_table_lite.Interpreter",
            return_value=interp,
        ):
            model.load_weights("/some/dir")

        assert model._tflite_output_index == 1

    def test_tensor_indices_from_details(self, model):
        interp = MagicMock()
        interp.get_input_details.return_value = [{"index": 7}]
        interp.get_output_details.return_value = [{"index": 42}]
        with patch(
            "corl.model.discrete.deep_value_table_lite.Interpreter",
            return_value=interp,
        ):
            model.load_weights("/some/dir")

        assert model._tflite_input_index == 7
        assert model._tflite_output_index == 42


# ---------------------------------------------------------------------------
# save_weights()
# ---------------------------------------------------------------------------


class TestSaveWeights:
    def test_always_raises_runtime_error(self, model):
        with pytest.raises(RuntimeError, match="ModelDeepValueTable"):
            model.save_weights("/out/dir")

    def test_raises_with_checkpoint_true(self, model):
        with pytest.raises(RuntimeError):
            model.save_weights("/out/dir", checkpoint=True)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPredict:
    def test_raises_when_interpreter_not_loaded(self, model, obs):
        with pytest.raises(RuntimeError):
            model.predict(obs)

    def test_returns_int32_scalar_array(self, loaded_model, obs):
        result = loaded_model.predict(obs)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32

    def test_returns_argmax_of_q_values(self):
        # q_values: action 2 has highest value
        q = np.array([[0.1, 0.3, 0.9, 0.0]], dtype=np.float32)
        m = make_loaded_model(q_values=q)
        obs = np.zeros((1, 8), dtype=np.float32)
        result = m.predict(obs)
        assert int(result) == 2

    def test_sets_input_tensor(self, loaded_model, obs):
        loaded_model.predict(obs)
        loaded_model._tflite_interpreter.set_tensor.assert_called_once_with(
            loaded_model._tflite_input_index, obs
        )

    def test_invokes_interpreter(self, loaded_model, obs):
        loaded_model.predict(obs)
        loaded_model._tflite_interpreter.invoke.assert_called_once()

    def test_reads_output_tensor(self, loaded_model, obs):
        loaded_model.predict(obs)
        loaded_model._tflite_interpreter.get_tensor.assert_called_once_with(
            loaded_model._tflite_output_index
        )

    def test_argmax_over_all_actions(self):
        # Ensure it picks the highest across all 6 actions
        q = np.array([[0.0, 0.1, 0.2, 0.3, 0.9, 0.4]], dtype=np.float32)
        m = make_loaded_model(q_values=q)
        obs = np.zeros((1, 8), dtype=np.float32)
        assert int(m.predict(obs)) == 4

    def test_first_action_selected_when_tied(self):
        q = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        m = make_loaded_model(q_values=q)
        obs = np.zeros((1, 8), dtype=np.float32)
        assert int(m.predict(obs)) == 0


# ---------------------------------------------------------------------------
# load_metadata()
# ---------------------------------------------------------------------------


class TestLoadMetadata:
    def test_is_noop(self, model):
        result = model.load_metadata("/some/dir")
        assert result is None

    def test_does_not_raise_on_missing_dir(self, model):
        model.load_metadata("/nonexistent/path")
