import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from corl.model.model import Model

logger = logging.getLogger(__name__)


class ModelActorCritic(Model):
    """
    Actor-Critic model backed by two Keras neural networks.

    The actor network maps observations to action logits (pre-softmax),
    while the critic network maps observations to scalar state-value
    estimates. Both networks can be independent architectures or share a
    common feature-extraction backbone externally.

    Two construction modes are supported:

    - **From existing Keras models**: supply ``actor``, ``critic``, and
      ``action_size``.
    - **From disk**: supply ``model_dir``; weights and metadata are loaded
      via :meth:`load`.

    Attributes:
        actor: Keras model that outputs action logits of shape
            ``(batch, action_size)``.
        critic: Keras model that outputs scalar value estimates of shape
            ``(batch, 1)``.
        action_size: Number of discrete actions in the actor output layer.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        actor: tf.keras.Model | None = None,
        critic: tf.keras.Model | None = None,
        action_size: int | None = None,
    ):
        """
        Initialise the actor-critic model.

        Exactly one construction mode must be used:

        - **From Keras models**: provide ``actor``, ``critic``, and
          ``action_size``. All three are required.
        - **From disk**: provide ``model_dir``. All other arguments are
          ignored.

        Args:
            model_dir: Directory containing a previously saved model. When
                provided, other arguments are ignored.
            actor: Pre-built Keras model outputting action logits.
            critic: Pre-built Keras model outputting scalar values.
            action_size: Number of discrete actions.

        Raises:
            ValueError: If ``model_dir`` is ``None`` and any of ``actor``,
                ``critic``, or ``action_size`` is not provided.
        """

        if model_dir is not None:
            if any(p is not None for p in (actor, critic, action_size)):
                logger.warning(
                    "model_dir provided alongside other arguments; "
                    "they will be ignored."
                )
            self.load(model_dir)
        else:
            if actor is None or critic is None or action_size is None:
                raise ValueError(
                    "actor, critic, and action_size must all be provided "
                    "when model_dir is not provided."
                )
            self.actor = actor
            self.critic = critic
            self.action_size = action_size

    def predict(
        self, observation: NDArray[np.float32]
    ) -> NDArray[np.int32] | NDArray[np.float32]:
        """
        Sample actions from the actor's policy distribution.

        Runs a forward pass through the actor network in inference mode
        (``training=False``), applies softmax to obtain action probabilities,
        and samples one action per observation from the resulting categorical
        distribution.

        Args:
            observation: Batch of observations, shape ``(batch_size, obs_dim)``.

        Returns:
            NDArray[np.int32]: Sampled action indices, shape ``(batch_size,)``.
        """
        logits = self.actor(observation, training=False)
        probs = tf.nn.softmax(logits).numpy()
        actions = np.array([np.random.choice(self.action_size, p=p) for p in probs])
        return actions.astype(np.int32)

    def save_weights(self, model_dir: str) -> None:
        """
        Persist actor and critic weights to ``model_dir``.

        Args:
            model_dir: Target directory for the weight files. Saves
                ``actor.keras`` and ``critic.keras`` in place.
        """
        base = Path(model_dir)
        self.actor.save(base / "actor.keras")
        self.critic.save(base / "critic.keras")
        logger.info(f"Actor-critic model saved to {model_dir}")

    def save_weights_lite(self, model_dir: str) -> None:
        """
        Convert and save the actor as TFLite to ``model_dir``.

        Uses ``tf.lite.TFLiteConverter`` to convert the actor Keras model
        to a TFLite flatbuffer. Only the actor is exported because the
        critic is not needed at inference time. The resulting file can be
        loaded by
        :class:`~corl.model.discrete.actor_critic_lite.ModelActorCriticLite` for
        lightweight inference.

        Args:
            model_dir: Target directory for ``actor.tflite``.
        """
        base = Path(model_dir)
        converter = tf.lite.TFLiteConverter.from_keras_model(self.actor)
        tflite_model = converter.convert()
        with open(base / "actor.tflite", "wb") as f:
            f.write(tflite_model)

        logger.info(f"Actor TFLite model saved to {model_dir}")

    def save(self, model_dir: str) -> None:
        """
        Save weights, metadata, and TFLite exports to ``model_dir``.

        Extends :meth:`~corl.model.model.Model.save` by additionally
        calling :meth:`save_weights_lite` so both ``.keras`` and
        ``.tflite`` files are written in a single call.

        Args:
            model_dir: Target directory for all output files.
        """
        super().save(model_dir)
        self.save_weights_lite(model_dir)

    def load_weights(self, model_dir: str) -> None:
        """
        Load actor and critic Keras models from ``model_dir``.

        Uses ``tf.keras.models.load_model`` which restores the full model
        (architecture, weights, and optimiser state), not just the raw
        weight tensors.

        Args:
            model_dir: Directory containing ``actor.keras`` and
                ``critic.keras``.

        Raises:
            FileNotFoundError: If either file does not exist.
        """
        base = Path(model_dir)
        self.actor = tf.keras.models.load_model(base / "actor.keras")
        self.critic = tf.keras.models.load_model(base / "critic.keras")
        logger.info(f"Loaded actor-critic model from {model_dir}")

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
