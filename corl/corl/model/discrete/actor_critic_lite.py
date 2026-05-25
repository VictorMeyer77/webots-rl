import json
import logging
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from numpy.typing import NDArray

from corl.model.model import Model

logger = logging.getLogger(__name__)


class ModelActorCriticLite(Model):
    """
    Inference-only actor model backed by an ai-edge-litert interpreter.

    Loads an ``actor.tflite`` file produced by
    :meth:`~corl.model.discrete.actor_critic.ModelActorCritic.save_weights_lite`
    and runs forward passes using the LiteRT runtime. The interpreter
    outputs action logits from which actions are sampled via softmax.

    Intended for deployment where the full TensorFlow training stack is
    not required. Only the actor is needed at inference time; the critic
    is used exclusively during training. Saving is not supported — use
    :class:`~corl.model.discrete.actor_critic.ModelActorCritic` to train and
    export the model, then load the resulting ``.tflite`` file here.

    Attributes:
        action_size: Number of discrete actions in the actor output layer.
        _actor_interpreter: LiteRT ``Interpreter`` for the actor network,
            or ``None`` until :meth:`load_weights` is called.
        _actor_input_index: Index of the actor's first input tensor.
        _actor_output_index: Index of the actor's first output tensor.
    """

    action_size: int

    _actor_interpreter: Interpreter | None
    _actor_input_index: int | None
    _actor_output_index: int | None

    def __init__(self, model_dir: str | None = None, action_size: int | None = None):
        """
        Initialise the lite actor model.

        Args:
            model_dir: Directory containing ``actor.tflite`` and
                ``metadata.json``. When provided, weights and metadata
                are loaded immediately via :meth:`load`.
            action_size: Number of discrete actions. Only used when
                ``model_dir`` is ``None`` (the value will be overridden
                by :meth:`load_metadata` when loading from disk).
        """
        super().__init__()
        self._actor_interpreter = None
        self._actor_input_index = None
        self._actor_output_index = None
        self.action_size = action_size or 0

        if model_dir is not None:
            self.load(model_dir)

    @staticmethod
    def _load_interpreter(
        model_path: Path,
    ) -> tuple[Interpreter, int, int]:
        """
        Create a LiteRT interpreter and return it with I/O tensor indices.

        Args:
            model_path: Path to the ``.tflite`` file.

        Returns:
            Tuple of ``(interpreter, input_index, output_index)``.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        interpreter = Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
        return interpreter, input_index, output_index

    def load_weights(self, model_dir: str) -> None:
        """
        Load the actor LiteRT model from ``model_dir``.

        Expects ``actor.tflite`` in the given directory. The interpreter
        is allocated and its I/O tensor indices are cached.

        Args:
            model_dir: Directory containing ``actor.tflite``.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        base = Path(model_dir)

        self._actor_interpreter, self._actor_input_index, self._actor_output_index = (
            self._load_interpreter(base / "actor.tflite")
        )

        logger.info(f"Loaded actor LiteRT model from {model_dir}")

    def save_weights(self, model_dir: str, checkpoint: bool = False) -> None:
        """
        Not supported — always raises ``RuntimeError``.

        LiteRT models are read-only at inference time. To produce
        ``.tflite`` files, call
        :meth:`~corl.model.discrete.actor_critic.ModelActorCritic.save` on the
        full Keras model.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(
            "LiteRT models should be saved with ModelActorCritic.save()."
        )

    def predict(
        self, observation: NDArray[np.float32]
    ) -> NDArray[np.int32] | NDArray[np.float32]:
        """
        Sample an action from the actor's policy distribution.

        Sets the actor's input tensor, invokes the interpreter, applies
        softmax to the output logits, and samples one action from the
        resulting categorical distribution.

        Args:
            observation: Observation array. Shape must match the actor
                model's expected input (typically ``(1, obs_dim)``).

        Returns:
            NDArray[np.int32]: Sampled action index (scalar array).

        Raises:
            RuntimeError: If :meth:`load_weights` has not been called.
        """
        if self._actor_interpreter is None:
            raise RuntimeError(
                "LiteRT actor interpreter not loaded. Call load_weights() first."
            )

        self._actor_interpreter.set_tensor(self._actor_input_index, observation)
        self._actor_interpreter.invoke()
        logits = self._actor_interpreter.get_tensor(self._actor_output_index)

        # Numerically stable softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        action = np.random.choice(self.action_size, p=probs.flatten())
        return np.array(action, dtype=np.int32)

    def load_metadata(self, model_dir: str) -> None:
        """
        Load ``action_size`` from ``<model_dir>/metadata.json``.

        Args:
            model_dir: Directory containing ``metadata.json``.

        Raises:
            FileNotFoundError: If ``metadata.json`` does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
            KeyError: If ``action_size`` is missing from the metadata.
        """

        with open(Path(model_dir) / "metadata.json", "r") as f:
            metadata = json.load(f)
        self.action_size = int(metadata["action_size"])
        logger.info(f"Loaded metadata: action_size={self.action_size}")
