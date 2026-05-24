import logging

import numpy as np
from numpy.typing import NDArray
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from corl.schemas.learning import Observation
from corl.schemas.tracker import StepKey
from corl.trainer.algorithm.continuous.ppo import TrainerPPOContinuous
from corl.utils.config import Config
from corl.utils.logger import setup_logging

ACTION_DIM = 2  # Continuous action space: [pitch_velocity, yaw_velocity]
TRANSITIONS = 1_000_000  # Total number of training transitions
CHECKPOINT_FREQUENCY = 400_000  # Save a checkpoint every N transitions
GAMMA = 0.99  # Discount factor
CLIP_RANGE = 0.2  # PPO clipping parameter ε
PPO_EPOCHS = 4  # Optimisation passes over each rollout
MINI_BATCH_SIZE = 64  # Transitions per mini-batch
ENTROPY_COEFF = 0.0  # Entropy bonus weight (Gaussian already explores)
VALUE_LOSS_COEFF = 0.5  # Critic MSE loss scaling factor
ACTOR_LR = 0.0003  # Actor Adam learning rate
CRITIC_LR = 0.0003  # Critic Adam learning rate
UPDATE_FREQUENCY = 2048  # Transitions collected between parameter updates
MAX_GRAD_NORM = 0.5  # L2 gradient clipping threshold
INPUT_SHAPE = 18  # Flat sensor vector size (3 accelerometers + 3 gyros) × 3 axes


def build_actor() -> Sequential:
    """
    Build the stochastic Gaussian actor network.

    Maps flat sensor observations to a vector of size ``ACTION_DIM * 2``
    with linear output activation, representing ``[mean, log_std]`` of the
    Gaussian policy. The ``TrainerPPOContinuous`` applies tanh squashing
    during action sampling.

    Returns:
        Keras ``Sequential`` model outputting ``ACTION_DIM * 2`` values.
    """
    model = Sequential(
        [
            Dense(256, activation="relu", input_shape=(INPUT_SHAPE,)),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(ACTION_DIM * 2, activation="linear"),
        ]
    )
    model.summary()
    return model


def build_critic() -> Sequential:
    """
    Build the state-value critic network.

    Maps flat sensor observations to a scalar state-value estimate.

    Returns:
        Keras ``Sequential`` model with scalar output.
    """
    model = Sequential(
        [
            Dense(256, activation="relu", input_shape=(INPUT_SHAPE,)),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )
    model.summary()
    return model


class PitEscapePPOContinuous(TrainerPPOContinuous):
    """
    Continuous PPO trainer for the pit escape task.

    Stochastic Gaussian policy outputting ``[pitch_velocity, yaw_velocity]``
    in the normalised range ``[-1, 1]`` via tanh squashing. Observations are
    the flat sensor vector from the BB-8 accelerometers and gyroscopes.
    """

    def parse_observations(
        self, observations: list[tuple[StepKey, Observation]]
    ) -> list[tuple[StepKey, NDArray[np.float32]]]:
        """
        Convert raw observations to float32 arrays for all workers.

        Args:
            observations: Raw observations received from all workers.

        Returns:
            List of ``(StepKey, flat_obs)`` pairs where each observation is a
            float32 array of shape ``(INPUT_SHAPE,)``.
        """
        return [
            (
                step_key,
                np.asarray(observation.data["base"], dtype=np.float32),
            )
            for step_key, observation in observations
        ]

    def params(self) -> dict[str, str | int | float]:
        """
        Return hyperparameters logged to the experiment tracker.

        Extends the parent params with the input shape specific to this trainer.

        Returns:
            Dict mapping parameter names to their values.
        """
        return super().params() | {"input_shape": INPUT_SHAPE}


if __name__ == "__main__":
    config = Config()
    setup_logging(config)
    logger = logging.getLogger(__name__)

    PitEscapePPOContinuous(
        config=config,
        actor=build_actor(),
        critic=build_critic(),
        action_dim=ACTION_DIM,
        checkpoint_frequency=CHECKPOINT_FREQUENCY,
        # checkpoint_id="20260524_222747",
        gamma=GAMMA,
        clip_range=CLIP_RANGE,
        ppo_epochs=PPO_EPOCHS,
        mini_batch_size=MINI_BATCH_SIZE,
        entropy_coeff=ENTROPY_COEFF,
        value_loss_coeff=VALUE_LOSS_COEFF,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        update_frequency=UPDATE_FREQUENCY,
        max_grad_norm=MAX_GRAD_NORM,
    ).run(max_transitions=TRANSITIONS)
