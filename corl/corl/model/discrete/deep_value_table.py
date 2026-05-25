import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from corl.model.model import Model

logger = logging.getLogger(__name__)


class ModelDeepValueTable(Model):
    """
    Deep Q-network model backed by a Keras neural network.

    Stores a ``tf.keras.Model`` that maps observation vectors to Q-value
    vectors of length ``action_size``. Greedy actions are obtained by taking
    the ``argmax`` over the Q-values. The model can be saved in both native
    Keras format (``model.keras``) and TFLite format (``model.tflite``) for
    deployment via :class:`~corl.model.deep_value_table_lite.ModelDeepValueTableLite`.

    Two construction modes are supported:

    - **From an existing Keras model**: supply ``weights`` and
      ``action_size``.
    - **From disk**: supply ``model_dir``; weights and metadata are loaded
      via :meth:`load`.

    Attributes:
        weights: The underlying ``tf.keras.Model`` used for inference and
            training.
        action_size: Number of discrete actions in the output layer.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        weights: tf.keras.Model | None = None,
        action_size: int | None = None,
    ):
        """
        Initialise the model, either from a Keras model or from disk.

        Exactly one construction mode must be used:

        - **From a Keras model**: provide ``weights`` and ``action_size``.
          Both are required; omitting either raises ``ValueError``.
        - **From disk**: provide ``model_dir``. Weights are loaded from
          ``model.keras`` and metadata from ``metadata.json`` via
          :meth:`load`. A warning is logged if ``weights`` or
          ``action_size`` are also supplied, as they are ignored.

        Args:
            model_dir: Directory containing a previously saved model. When
                provided, ``weights`` and ``action_size`` are ignored.
            weights: Pre-built ``tf.keras.Model`` to use directly. Required
                when ``model_dir`` is ``None``.
            action_size: Number of discrete actions. Required when
                ``model_dir`` is ``None``.

        Raises:
            ValueError: If ``model_dir`` is ``None`` and either ``weights``
                or ``action_size`` is not provided.
        """

        if model_dir is not None:
            if any(p is not None for p in (action_size, weights)):
                logger.warning(
                    "model_dir provided alongside dimension arguments; dimension arguments will be ignored."
                )
            self.load(model_dir)
        else:
            if weights is None or action_size is None:
                raise ValueError(
                    "weights and action_size must be provided when model_dir is not provided."
                )

            self.weights = weights
            self.action_size = action_size

    def load_weights(self, model_dir: str) -> None:
        """
        Load the Keras model from ``<model_dir>/model.keras``.

        Args:
            model_dir: Directory containing ``model.keras``.

        Raises:
            FileNotFoundError: If ``model.keras`` does not exist.
            ValueError: If the file cannot be parsed as a valid Keras model.
        """
        model_path = Path(model_dir) / "model.keras"
        self.weights = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded TensorFlow model from {model_path}")

    def save_weights(self, model_dir: str) -> None:
        """
        Persist the Keras model to ``<model_dir>/model.keras``.

        Args:
            model_dir: Target directory for the ``.keras`` file.
        """

        model_path = Path(model_dir) / "model.keras"
        self.weights.save(model_path)
        logger.info(f"Model saved: {model_path}.")

    def save_weights_lite(self, model_dir: str) -> None:
        """
        Convert and save the model as TFLite to ``<model_dir>/model.tflite``.

        Uses ``tf.lite.TFLiteConverter`` to convert the Keras model to a
        TFLite flatbuffer. The resulting file can be loaded by
        :class:`~corl.model.deep_value_table_lite.ModelDeepValueTableLite`
        for lightweight inference.

        Args:
            model_dir: Target directory for ``model.tflite``.
        """

        model_path = Path(model_dir) / "model.tflite"
        converter = tf.lite.TFLiteConverter.from_keras_model(self.weights)
        tflite_model = converter.convert()

        with open(model_path, "wb") as f:
            f.write(tflite_model)

    def save(self, model_dir: str) -> None:
        """
        Save weights, metadata, and TFLite export to ``model_dir``.

        Extends :meth:`~corl.model.model.Model.save` by additionally calling
        :meth:`save_weights_lite` so both ``model.keras`` and
        ``model.tflite`` are written in a single call.

        Args:
            model_dir: Target directory for all output files.
        """
        super().save(model_dir)
        self.save_weights_lite(model_dir)

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

    def predict(
        self, observation: NDArray[np.float32]
    ) -> NDArray[np.int32] | NDArray[np.float32]:
        """
        Return the greedy action for each observation in the batch.

        Runs a forward pass through :attr:`weights` with ``training=False``
        and returns the ``argmax`` over Q-values along the action axis.

        Args:
            observation: Batch of observations, shape ``(batch_size, obs_dim)``.

        Returns:
            NDArray[np.int32]: Greedy action indices, shape ``(batch_size,)``.

        Raises:
            ValueError: If :attr:`weights` has not been initialised.
        """

        if self.weights is None:
            raise ValueError("model is not initialized.")

        q_values = self.weights(observation, training=False).numpy()
        return np.argmax(q_values, axis=-1).astype(np.int32)

    def epsilon_greedy_policy(
        self, observation: NDArray[np.float32], epsilon: float
    ) -> NDArray[np.int32]:
        """
        Select actions for a batch of observations using an ε-greedy policy.

        Each sample in the batch independently explores (random action) with
        probability ``epsilon`` or exploits (greedy action from Q-values)
        with probability ``1 - epsilon``.

        Args:
            observation: Batch of observations, shape ``(batch_size, obs_dim)``.
            epsilon: Exploration rate in ``[0, 1]``. ``0`` is fully greedy;
                ``1`` is fully random.

        Returns:
            NDArray[np.int32]: Selected action indices, shape ``(batch_size,)``.
        """

        batch_size = observation.shape[0]
        explore_mask = np.random.random(batch_size) < epsilon

        q_values = self.weights(
            observation, training=False
        ).numpy()  # (batch, action_size)
        greedy_actions = np.argmax(q_values, axis=-1)  # (batch,)
        random_actions = np.random.randint(self.action_size, size=batch_size)

        actions = np.where(explore_mask, random_actions, greedy_actions)
        logger.debug(
            f"{explore_mask.sum()} of {batch_size} elements taking random action"
        )
        return actions.astype(np.int32)
