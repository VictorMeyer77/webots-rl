"""
Unit tests for corl.model.model.Model

Model is abstract so every test uses a minimal concrete subclass.
No real file system access is required — file I/O is patched throughout.
"""

import json
import os
from typing import Any
from unittest.mock import mock_open, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from corl.model.model import Model

# ---------------------------------------------------------------------------
# Minimal concrete subclass
# ---------------------------------------------------------------------------


class ConcreteModel(Model):
    """Minimal implementation of Model for testing purposes."""

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.checkpoint_index: int = 0
        self.save_weights_calls: list[tuple[str, bool]] = []
        self.load_weights_calls: list[str] = []

    def predict(self, observation: NDArray[np.float32]) -> NDArray[np.int32]:
        return np.zeros(1, dtype=np.int32)

    def save_weights(self, model_dir: str, checkpoint: bool = False) -> None:
        self.save_weights_calls.append((model_dir, checkpoint))

    def load_weights(self, model_dir: str) -> None:
        self.load_weights_calls.append(model_dir)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> ConcreteModel:
    return ConcreteModel()


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_concrete_subclass_instantiates(self, model):
        assert isinstance(model, Model)

    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Model()  # type: ignore

    def test_default_checkpoint_index(self, model):
        assert model.checkpoint_index == 0

    def test_default_metadata_is_instance_dict(self, model):
        """Each instance should have its own metadata dict, not the shared class default."""
        other = ConcreteModel()
        model.metadata["key"] = "value"
        assert "key" not in other.metadata


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPredict:
    def test_returns_array(self, model):
        obs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = model.predict(obs)
        assert isinstance(result, np.ndarray)

    def test_returns_zeros_for_concrete_stub(self, model):
        obs = np.zeros(4, dtype=np.float32)
        result = model.predict(obs)
        np.testing.assert_array_equal(result, np.zeros(1, dtype=np.int32))


# ---------------------------------------------------------------------------
# set_metadata()
# ---------------------------------------------------------------------------


class TestSetMetadata:
    def test_replaces_metadata(self, model):
        model.set_metadata({"a": 1, "b": 2})
        assert model.metadata == {"a": 1, "b": 2}

    def test_overwrites_existing_metadata(self, model):
        model.metadata = {"old": True}
        model.set_metadata({"new": True})
        assert model.metadata == {"new": True}
        assert "old" not in model.metadata

    def test_set_metadata_assigns_instance_dict(self):
        """set_metadata must assign to the instance, not the class-level default."""
        m1 = ConcreteModel()
        m2 = ConcreteModel()
        m1.set_metadata({"x": 1})
        assert "x" not in m2.metadata


# ---------------------------------------------------------------------------
# save_metadata() / load_metadata()
# ---------------------------------------------------------------------------


class TestSaveMetadata:
    def test_writes_json_to_correct_path(self, model):
        model.metadata = {"epoch": 5, "loss": 0.42}
        m = mock_open()
        with patch("builtins.open", m):
            model.save_metadata("/some/dir")

        m.assert_called_once_with(os.path.join("/some/dir", "metadata.json"), "w")
        handle = m()
        written = "".join(call.args[0] for call in handle.write.call_args_list)
        parsed = json.loads(written)
        assert parsed == {"epoch": 5, "loss": 0.42}

    def test_raises_oserror_on_write_failure(self, model):
        with patch("builtins.open", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                model.save_metadata("/bad/dir")


class TestLoadMetadata:
    def test_loads_json_from_correct_path(self, model):
        payload = json.dumps({"epoch": 3, "score": 0.9})
        m = mock_open(read_data=payload)
        with patch("builtins.open", m):
            model.load_metadata("/some/dir")

        m.assert_called_once_with(os.path.join("/some/dir", "metadata.json"), "r")
        assert model.metadata == {"epoch": 3, "score": 0.9}

    def test_raises_file_not_found(self, model):
        with patch("builtins.open", side_effect=FileNotFoundError("no file")):
            with pytest.raises(FileNotFoundError):
                model.load_metadata("/missing/dir")

    def test_raises_json_decode_error_on_malformed_file(self, model):
        m = mock_open(read_data="not valid json {{{")
        with patch("builtins.open", m):
            with pytest.raises(json.JSONDecodeError):
                model.load_metadata("/bad/dir")


# ---------------------------------------------------------------------------
# save() / load()
# ---------------------------------------------------------------------------


class TestSave:
    def test_calls_save_weights_then_save_metadata(self, model):
        call_order: list[str] = []
        model.save_weights = lambda d, **kw: call_order.append("weights")  # type: ignore
        with patch.object(
            model, "save_metadata", side_effect=lambda d: call_order.append("metadata")
        ):
            model.save("/out/dir")

        assert call_order == ["weights", "metadata"]

    def test_passes_correct_dir_to_save_weights(self, model):
        with patch.object(model, "save_metadata"):
            model.save("/out/dir")
        assert model.save_weights_calls == [("/out/dir", False)]

    def test_passes_correct_dir_to_save_metadata(self, model):
        with patch.object(model, "save_metadata") as mock_meta:
            model.save("/out/dir")
        mock_meta.assert_called_once_with("/out/dir")

    def test_does_not_pass_checkpoint_flag(self, model):
        """save() is a non-checkpoint convenience; checkpoint should be False."""
        with patch.object(model, "save_metadata"):
            model.save("/out/dir")
        _, checkpoint = model.save_weights_calls[0]
        assert checkpoint is False


class TestLoad:
    def test_calls_load_weights_then_load_metadata(self, model):
        call_order: list[str] = []
        model.load_weights = lambda d: call_order.append("weights")  # type: ignore
        with patch.object(
            model, "load_metadata", side_effect=lambda d: call_order.append("metadata")
        ):
            model.load("/in/dir")

        assert call_order == ["weights", "metadata"]

    def test_passes_correct_dir_to_load_weights(self, model):
        with patch.object(model, "load_metadata"):
            model.load("/in/dir")
        assert model.load_weights_calls == ["/in/dir"]

    def test_passes_correct_dir_to_load_metadata(self, model):
        with patch.object(model, "load_metadata") as mock_meta:
            model.load("/in/dir")
        mock_meta.assert_called_once_with("/in/dir")


# ---------------------------------------------------------------------------
# save_weights() / load_weights() (abstract contract via concrete stub)
# ---------------------------------------------------------------------------


class TestAbstractMethods:
    def test_save_weights_records_call(self, model):
        model.save_weights("/w/dir", checkpoint=True)
        assert model.save_weights_calls == [("/w/dir", True)]

    def test_load_weights_records_call(self, model):
        model.load_weights("/w/dir")
        assert model.load_weights_calls == ["/w/dir"]

    def test_subclass_missing_predict_raises(self):
        class Incomplete(Model):
            def save_weights(self, model_dir, checkpoint=False): ...
            def load_weights(self, model_dir): ...

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore

    def test_subclass_missing_save_weights_raises(self):
        class Incomplete(Model):
            def predict(self, observation): ...
            def load_weights(self, model_dir): ...

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore

    def test_subclass_missing_load_weights_raises(self):
        class Incomplete(Model):
            def predict(self, observation): ...
            def save_weights(self, model_dir, checkpoint=False): ...

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore
