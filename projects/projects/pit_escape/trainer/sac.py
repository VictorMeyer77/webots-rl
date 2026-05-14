import logging

import numpy as np
from numpy.typing import NDArray
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential

from corl.schemas.learning import Observation
from corl.schemas.tracker import StepKey
from corl.trainer.algorithm.continuous.sac import TrainerSAC
from corl.utils.config import Config
from corl.utils.logger import setup_logging

ACTION_DIM = 2  # Continuous action space: [pitch_velocity, yaw_velocity]
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
INPUT_SHAPE = 18  # Flat sensor vector size (3 accelerometers + 3 gyros) × 3 axes


def build_actor() -> Sequential:
    """
    Build the Gaussian actor network.

    Maps flat sensor observations to a vector of size ``ACTION_DIM * 2``:
    the first half is the action mean and the second half is the log-std.
    The SAC trainer applies the reparameterisation trick and ``tanh``
    squashing at training time; the exported TFLite model outputs the
    mean directly for inference.

    Returns:
        Keras ``Sequential`` model outputting ``ACTION_DIM * 2`` values.
    """
    model = Sequential(
        [
            Dense(128, activation="relu", input_shape=(INPUT_SHAPE,)),
            Dense(64, activation="relu"),
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
        ``(INPUT_SHAPE + ACTION_DIM,)`` and scalar output.
    """
    obs_input = Input(shape=(INPUT_SHAPE + ACTION_DIM,))
    x = Dense(128, activation="relu")(obs_input)
    x = Dense(64, activation="relu")(x)
    q_value = Dense(1, activation="linear")(x)
    model = Model(inputs=obs_input, outputs=q_value)
    model.summary()
    return model


class PitEscapeSAC(TrainerSAC):
    """
    SAC trainer for the pit escape task.

    Continuous-action policy outputting ``[pitch_velocity, yaw_velocity]``
    in the normalised range ``[-1, 1]``. Observations are the flat sensor
    vector from the BB-8 accelerometers and gyroscopes.
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

    PitEscapeSAC(
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
