import logging

import numpy as np
from numpy.typing import NDArray
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential

from corl.schemas.learning import Observation
from corl.schemas.tracker import StepKey
from corl.trainer.algorithm.continuous.td3 import TrainerTD3
from corl.utils.config import Config
from corl.utils.logger import setup_logging

ACTION_DIM = 2  # Continuous action space: [pitch_velocity, yaw_velocity]
TRANSITIONS = 1_000_000  # Total number of training transitions
CHECKPOINT_FREQUENCY = 400_000  # Save a checkpoint every N transitions
GAMMA = 0.99  # Discount factor
TAU = 0.005  # Soft target update coefficient
BATCH_SIZE = 256  # Transitions sampled per gradient step
FIT_FREQUENCY = 1  # Steps between gradient updates
ACTOR_LR = 0.0003  # Actor Adam learning rate
CRITIC_LR = 0.0001  # Critic Adam learning rate
POLICY_DELAY = 2  # Critic updates per actor update
EXPLORATION_NOISE = 0.1  # Std of Gaussian noise added to actions during rollout
TARGET_NOISE = 0.2  # Std of smoothing noise added to target actions
TARGET_NOISE_CLIP = 0.5  # Absolute clip bound for target smoothing noise
MAX_GRAD_NORM = 1.0  # L2 gradient clipping threshold
PER_SIZE = 300_000  # Replay buffer capacity
PER_ALPHA = 0.6  # PER priority exponent
PER_BETA_START = 0.4  # Initial IS correction exponent
INPUT_SHAPE = 18  # Flat sensor vector size (3 accelerometers + 3 gyros)  3 axes


def build_actor() -> Sequential:
    """
    Build the deterministic actor network.

    Maps flat sensor observations to a vector of size ``ACTION_DIM`` with
    ``tanh`` output activation, producing actions directly in ``[-1, 1]``.

    Returns:
        Keras ``Sequential`` model outputting ``ACTION_DIM`` values.
    """
    model = Sequential(
        [
            Dense(256, activation="relu", input_shape=(INPUT_SHAPE,)),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(ACTION_DIM, activation="tanh"),
        ]
    )
    model.summary()
    return model


def build_critic() -> Model:
    """
    Build a Q-network critic.

    Accepts a concatenated ``(observation, action)`` input and outputs a
    scalar Q-value. Two independent instances are created for the twin-
    critic architecture used by TD3.

    Returns:
        Keras functional ``Model`` with input shape
        ``(INPUT_SHAPE + ACTION_DIM,)`` and scalar output.
    """
    obs_input = Input(shape=(INPUT_SHAPE + ACTION_DIM,))
    x = Dense(256, activation="relu")(obs_input)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    q_value = Dense(1, activation="linear")(x)
    model = Model(inputs=obs_input, outputs=q_value)
    model.summary()
    return model


class PitEscapeTD3(TrainerTD3):
    """
    TD3 trainer for the pit escape task.

    Deterministic continuous-action policy outputting
    ``[pitch_velocity, yaw_velocity]`` in the normalised range ``[-1, 1]``.
    Observations are the flat sensor vector from the BB-8 accelerometers
    and gyroscopes. Gaussian exploration noise is added during rollout.
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

    PitEscapeTD3(
        config=config,
        actor=build_actor(),
        critic1=build_critic(),
        critic2=build_critic(),
        action_dim=ACTION_DIM,
        checkpoint_frequency=CHECKPOINT_FREQUENCY,
        # checkpoint_id="20260524_222240",
        gamma=GAMMA,
        tau=TAU,
        batch_size=BATCH_SIZE,
        fit_frequency=FIT_FREQUENCY,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        policy_delay=POLICY_DELAY,
        exploration_noise=EXPLORATION_NOISE,
        target_noise=TARGET_NOISE,
        target_noise_clip=TARGET_NOISE_CLIP,
        max_grad_norm=MAX_GRAD_NORM,
        per_size=PER_SIZE,
        per_alpha=PER_ALPHA,
        per_beta_start=PER_BETA_START,
    ).run(max_transitions=TRANSITIONS)
