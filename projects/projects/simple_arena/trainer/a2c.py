import logging

import numpy as np
from numpy.typing import NDArray
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from corl.model.discrete.actor_critic import ModelActorCritic
from corl.schemas.learning import Observation
from corl.schemas.tracker import StepKey
from corl.trainer.algorithm.discrete.a2c import TrainerA2C
from corl.utils.config import Config
from corl.utils.logger import setup_logging

ACTION_SIZE = 9  # Number of discrete actions available to the agent
TRANSITIONS = 1_000_000  # Total number of training transitions
MODEL_CHECKPOINT_FREQUENCY = 250_000  # Save a checkpoint every N transitions
GAMMA = 0.99  # Discount factor: how much future rewards are valued
ENTROPY_COEFF = 0.1  # Entropy bonus weight to encourage exploration
VALUE_LOSS_COEFF = 0.5  # Critic loss scaling factor
ACTOR_LR = 1e-4  # Actor Adam learning rate
CRITIC_LR = 1e-4  # Critic Adam learning rate
UPDATE_FREQUENCY = 64  # Transitions to collect between A2C updates
OBS_SIZE = 42 * 42 * 4  # Flattened input size (height * width * stacked frames)


def build_actor() -> Sequential:
    """
    Build the actor network.

    Maps observations to action logits (pre-softmax). The softmax
    sampling is applied inside :class:`~corl.model.actor_critic.ModelActorCritic`.

    Returns:
        Keras ``Sequential`` model outputting ``ACTION_SIZE`` logits.
    """
    model = Sequential(
        [
            Dense(256, activation="relu", input_shape=(OBS_SIZE,)),
            Dense(128, activation="relu"),
            Dense(ACTION_SIZE, activation="linear"),
        ]
    )
    model.summary()
    return model


def build_critic() -> Sequential:
    """
    Build the critic network.

    Maps observations to a scalar state-value estimate.

    Returns:
        Keras ``Sequential`` model outputting a single value.
    """
    model = Sequential(
        [
            Dense(256, activation="relu", input_shape=(OBS_SIZE,)),
            Dense(128, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )
    model.summary()
    return model


class SimpleArenaA2C(TrainerA2C):
    """
    A2C trainer for the simple arena task.

    Uses separate actor and critic networks with dense layers.
    Observations are raw camera frames flattened before being fed to
    the networks.
    """

    def parse_observations(
        self, observations: list[tuple[StepKey, Observation]]
    ) -> list[tuple[StepKey, NDArray[np.float32]]]:
        """
        Flatten camera frames for all workers.

        Args:
            observations: Raw observations received from all workers in the
                current rollout.

        Returns:
            List of ``(StepKey, flat_frame)`` pairs where each frame is a
            float32 array of shape ``(OBS_SIZE,)``.
        """
        return [
            (
                step_key,
                np.asarray(observation.data["camera"], dtype=np.float32).flatten(),
            )
            for step_key, observation in observations
        ]

    def params(self) -> dict[str, str | int | float]:
        """
        Return hyperparameters logged to the experiment tracker.

        Extends the parent params with the observation size specific to
        this trainer.

        Returns:
            Dict mapping parameter names to their values.
        """
        return super().params() | {"obs_size": OBS_SIZE}


if __name__ == "__main__":
    config = Config()
    setup_logging(config)
    logger = logging.getLogger(__name__)

    model = ModelActorCritic(
        actor=build_actor(),
        critic=build_critic(),
        action_size=ACTION_SIZE,
    )

    SimpleArenaA2C(
        config=config,
        model=model,
        model_checkpoint_frequency=MODEL_CHECKPOINT_FREQUENCY,
        gamma=GAMMA,
        entropy_coeff=ENTROPY_COEFF,
        value_loss_coeff=VALUE_LOSS_COEFF,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        update_frequency=UPDATE_FREQUENCY,
    ).run(epochs=TRANSITIONS)
