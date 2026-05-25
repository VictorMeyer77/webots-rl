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
    and runs forward passes using the LiteRT runtime.
    Note: :class:`~corl.model.discrete.actor_critic.ModelActorCritic` is the
    shared training class used by both discrete and continuous algorithms.

    Supports SAC and TD3 actor architectures:

    - **SAC**: the actor outputs ``action_size * 2`` values encoding
      ``[mean, log_std]`` of a Gaussian policy; ``predict`` returns
      ``tanh(mean)`` as a deterministic float32 action.
    - **TD3**: the actor outputs ``action_size`` values with ``tanh``
      already applied; ``predict`` returns the raw output directly.

    Intended for deployment where the full TensorFlow training stack is
    not required. Only the actor is needed at inference time; the critic
    is used exclusively during training. Saving is not supported — train
    and export the model with
    :class:`~corl.model.discrete.actor_critic.ModelActorCritic`, then load
    the resulting ``.tflite`` file here.

    Attributes:
        action_size: Number of action dimensions.
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
        Run a forward pass and return a continuous action.

        Behaviour depends on the actor output shape:

        - **SAC** (output size == ``action_size * 2``): the output encodes
          ``[mean, log_std]`` from a Gaussian policy. The deterministic mean
          is extracted and squashed through ``tanh`` to return a float32
          array of shape ``(action_size,)``.
        - **TD3** (output size == ``action_size``): the actor already applies
          ``tanh`` as its output activation; the raw output is returned
          directly as a float32 array of shape ``(action_size,)``.

        Args:
            observation: Observation array. Shape must match the actor
                model's expected input (typically ``(1, obs_dim)``).

        Returns:
            ``NDArray[np.float32]`` of shape ``(action_size,)`` with values
            in ``[-1, 1]``.

        Raises:
            RuntimeError: If :meth:`load_weights` has not been called.
        """
        if self._actor_interpreter is None:
            raise RuntimeError(
                "LiteRT actor interpreter not loaded. Call load_weights() first."
            )

        self._actor_interpreter.set_tensor(self._actor_input_index, observation)
        self._actor_interpreter.invoke()
        output = self._actor_interpreter.get_tensor(self._actor_output_index)

        if output.shape[-1] == self.action_size * 2:
            # SAC Gaussian policy: output is [mean, log_std]; use tanh(mean)
            mean = output[..., : self.action_size]
            return np.tanh(mean).astype(np.float32).flatten()

        # TD3 deterministic policy: tanh already applied by the model
        return output.astype(np.float32).flatten()

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
