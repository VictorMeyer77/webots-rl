import logging

import numpy as np
from numpy.typing import NDArray
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential

from corl.schemas.learning import Observation
from corl.schemas.tracker import StepKey
from corl.trainer.algorithm.discrete.continuous.sac import TrainerSAC
from corl.utils.config import Config
from corl.utils.logger import setup_logging

ACTION_DIM = 2  # Continuous action space: [left_velocity, right_velocity]
TRANSITIONS = 1_000_000  # Total number of training transitions
MODEL_CHECKPOINT_FREQUENCY = 250_000  # Save a checkpoint every N transitions
GAMMA = 0.99  # Discount factor
TAU = 0.005  # Soft target update coefficient
BATCH_SIZE = 256  # Transitions sampled per gradient step
FIT_FREQUENCY = 1  # Steps between gradient updates
ACTOR_LR = 3e-4  # Actor Adam learning rate
CRITIC_LR = 3e-4  # Critic Adam learning rate
ALPHA = 0.2  # Initial entropy temperature
AUTO_ALPHA = True  # Automatically tune entropy temperature
PER_SIZE = 100_000  # Replay buffer capacity
PER_ALPHA = 0.6  # PER priority exponent
PER_BETA_START = 0.4  # Initial IS correction exponent
OBS_SIZE = 42 * 42 * 4  # Flattened input size (height * width * stacked frames)


def build_actor() -> Sequential:
    """
    Build the Gaussian actor network.

    Maps observations to a flat vector of size ``ACTION_DIM * 2``:
    the first half is the action mean and the second half is the log-std.
    The SAC trainer applies the reparameterisation trick and ``tanh``
    squashing at training time; the exported TFLite model outputs the
    mean directly for inference.

    Returns:
        Keras ``Sequential`` model outputting ``ACTION_DIM * 2`` values.
    """
    model = Sequential(
        [
            Dense(256, activation="relu", input_shape=(OBS_SIZE,)),
            Dense(256, activation="relu"),
            Dense(ACTION_DIM * 2, activation="linear"),
        ]
    )
    model.summary()
    return model


def build_critic() -> Model:
    """
    Build a Q-network critic.

    Accepts a concatenated ``(observation, action)`` input and outputs a
    scalar Q-value. Two independent instances are created for the twin-
    critic architecture used by SAC.

    Returns:
        Keras functional ``Model`` with input shape
        ``(OBS_SIZE + ACTION_DIM,)`` and scalar output.
    """
    obs_input = Input(shape=(OBS_SIZE + ACTION_DIM,))
    x = Dense(256, activation="relu")(obs_input)
    x = Dense(256, activation="relu")(x)
    q_value = Dense(1, activation="linear")(x)
    model = Model(inputs=obs_input, outputs=q_value)
    model.summary()
    return model


class SimpleArenaSAC(TrainerSAC):
    """
    SAC trainer for the simple arena task.

    Continuous-action policy outputting ``[left_velocity, right_velocity]``
    directly. Observations are raw camera frames flattened before being fed
    to the actor and twin-critic networks.
    """

    def parse_observations(
        self, observations: list[tuple[StepKey, Observation]]
    ) -> list[tuple[StepKey, NDArray[np.float32]]]:
        """
        Flatten camera frames for all workers.

        Args:
            observations: Raw observations received from all workers.

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

        Extends the parent params with the flattened observation size specific
        to this trainer.

        Returns:
            Dict mapping parameter names to their values.
        """
        return super().params() | {"obs_size": OBS_SIZE}


if __name__ == "__main__":
    config = Config()
    setup_logging(config)
    logger = logging.getLogger(__name__)

    SimpleArenaSAC(
        config=config,
        actor=build_actor(),
        critic1=build_critic(),
        critic2=build_critic(),
        action_dim=ACTION_DIM,
        model_checkpoint_frequency=MODEL_CHECKPOINT_FREQUENCY,
        gamma=GAMMA,
        tau=TAU,
        batch_size=BATCH_SIZE,
        fit_frequency=FIT_FREQUENCY,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        alpha=ALPHA,
        auto_alpha=AUTO_ALPHA,
        per_size=PER_SIZE,
        per_alpha=PER_ALPHA,
        per_beta_start=PER_BETA_START,
    ).run(epochs=TRANSITIONS)
