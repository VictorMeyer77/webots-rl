import logging
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from numpy.typing import NDArray

from corl.model.model import Model

logger = logging.getLogger(__name__)


class ModelDeepValueTableLite(Model):
    """
    Inference-only Deep Q-network backed by an ai-edge-litert interpreter.

    Loads a ``model.tflite`` file produced by
    :meth:`~corl.model.deep_value_table.ModelDeepValueTable.save_weights_lite`
    and runs forward passes using the LiteRT runtime. Intended for
    deployment where the full TensorFlow training stack is not required.

    Saving is not supported — use
    :class:`~corl.model.deep_value_table.ModelDeepValueTable` to train and
    export the model, then load the resulting ``.tflite`` file here.

    Attributes:
        _tflite_interpreter: The LiteRT ``Interpreter`` instance, or
            ``None`` until :meth:`load_weights` is called.
        _tflite_input_index: Index of the first input tensor, cached after
            ``allocate_tensors()``.
        _tflite_output_index: Index of the first output tensor, cached after
            ``allocate_tensors()``.
    """

    _tflite_interpreter: Interpreter | None = None
    _tflite_input_index: int | None = None
    _tflite_output_index: int | None = None

    def load_weights(self, model_dir: str):
        """
        Load a LiteRT model from ``<model_dir>/model.tflite``.

        Creates the LiteRT interpreter, allocates tensors, and caches the
        input and output tensor indices for use in :meth:`predict`.

        Args:
            model_dir: Directory containing ``model.tflite``.

        Raises:
            FileNotFoundError: If ``model.tflite`` does not exist in
                ``model_dir``.
        """
        model_path = Path(model_dir) / "model.tflite"
        self._tflite_interpreter = Interpreter(model_path=str(model_path))
        self._tflite_interpreter.allocate_tensors()
        self._tflite_input_index = self._tflite_interpreter.get_input_details()[0][
            "index"
        ]
        self._tflite_output_index = self._tflite_interpreter.get_output_details()[0][
            "index"
        ]

    def save_weights(self, model_dir: str, checkpoint: bool = False) -> None:
        """
        Not supported — always raises ``RuntimeError``.

        LiteRT models are read-only at inference time. To produce a
        ``.tflite`` file, call
        :meth:`~corl.model.deep_value_table.ModelDeepValueTable.save`
        on the full Keras model.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(
            "LiteRT model should be saved with ModelDeepValueTable.save()."
        )

    def predict(
        self, observation: NDArray[np.float32]
    ) -> NDArray[np.int32] | NDArray[np.float32]:
        """
        Return the greedy action for the given observation.

        Sets the input tensor, invokes the LiteRT interpreter, and returns
        the ``argmax`` over the output Q-values.

        Args:
            observation: Observation array. Shape must match the model's
                expected input tensor (typically ``(1, obs_dim)`` for a
                single step or ``(batch_size, obs_dim)`` for a batch).

        Returns:
            NDArray[np.int32]: Greedy action index (scalar array).

        Raises:
            RuntimeError: If :meth:`load_weights` has not been called.
        """

        if self._tflite_interpreter is None:
            raise RuntimeError(
                "LiteRT interpreter not loaded. Call load_weights() first."
            )
        self._tflite_interpreter.set_tensor(self._tflite_input_index, observation)
        self._tflite_interpreter.invoke()
        q_values = self._tflite_interpreter.get_tensor(self._tflite_output_index)
        return np.array(np.argmax(q_values), dtype=np.int32)

    def load_metadata(self, model_dir: str) -> None:
        return
