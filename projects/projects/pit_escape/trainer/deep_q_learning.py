import logging

import numpy as np
from numpy.typing import NDArray
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from corl.model.discrete.deep_value_table import ModelDeepValueTable
from corl.schemas.learning import Observation
from corl.schemas.tracker import StepKey
from corl.trainer.algorithm.discrete.deep_q_learning import TrainerDeepQLearning
from corl.utils.config import Config
from corl.utils.logger import setup_logging

ACTION_SIZE = 9  # Number of discrete actions available to the agent
TRANSITIONS = 2_000_000  # Total number of training transitions
CHECKPOINT_FREQUENCY = 400_000  # Save a checkpoint every N transitions
GAMMA = 0.99  # Discount factor: how much future rewards are valued
EPSILON = 1.0  # Epsilon-greedy initial value
EPSILON_MIN = 0.01  # Epsilon-greedy minimum value
EPSILON_DECAY = 0.999995  # Epsilon-greedy decay rate per transition (reaches EPSILON_MIN at ~920k steps)
BATCH_SIZE = 64  # Number of transitions sampled per gradient update
FIT_FREQUENCY = 1  # Train the online network every N steps
UPDATE_TARGET_WEIGHTS_FREQUENCY = 2500  # Sync target network weights every N steps
PER_SIZE = 300_000  # Prioritised replay buffer capacity
PER_ALPHA = 0.6  # Prioritisation exponent (0 = uniform, 1 = full priority)
PER_BETA_START = 0.4  # Initial importance-sampling correction exponent
LEARNING_RATE = 0.0001  # Adam optimizer learning rate
INPUT_SHAPE = 18  # Input shape


def build_tf_model() -> Sequential:
    """
    Build and compile the Q-network for flat sensor input.

    The architecture maps a flat observation vector through two dense hidden
    layers to one Q-value per action.

    Returns:
        Compiled Keras ``Sequential`` model ready for training.
    """
    model = Sequential()
    model.add(Dense(256, activation="relu", input_shape=(INPUT_SHAPE,)))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(ACTION_SIZE, activation="linear"))
    model.compile(loss=Huber(), optimizer=Adam(learning_rate=LEARNING_RATE))
    model.summary()
    return model


class PitEscapeDeepQLearning(TrainerDeepQLearning):
    """
    Deep Q-learning trainer for the pit escape task.

    Uses a dense Q-network with a prioritised experience replay buffer
    and a target network updated periodically. Observations are flat
    sensor vectors of shape ``(INPUT_SHAPE,)`` passed directly to the network.
    """

    def parse_observations(
        self, observations: list[tuple[StepKey, Observation]]
    ) -> list[tuple[StepKey, NDArray[np.float32]]]:
        """
        Convert raw observations to float32 arrays for all workers.

        Args:
            observations: Raw observations received from all workers in the
                current rollout.

        Returns:
            List of ``(StepKey, flat_obs)`` pairs for every worker, where each
            observation is a float32 array of shape ``(INPUT_SHAPE,)``.
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

        Extends the parent params with the learning rate and input shape
        specific to this trainer.

        Returns:
            Dict mapping parameter names to their values.
        """
        return super().params() | {
            "learning_rate": LEARNING_RATE,
            "input_shape": INPUT_SHAPE,
        }


if __name__ == "__main__":
    config = Config()
    setup_logging(config)
    logger = logging.getLogger(__name__)

    model = ModelDeepValueTable(weights=build_tf_model(), action_size=ACTION_SIZE)

    PitEscapeDeepQLearning(
        config=config,
        model=model,
        checkpoint_frequency=CHECKPOINT_FREQUENCY,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        fit_frequency=FIT_FREQUENCY,
        update_target_weights_frequency=UPDATE_TARGET_WEIGHTS_FREQUENCY,
        per_size=PER_SIZE,
        per_alpha=PER_ALPHA,
        per_beta_start=PER_BETA_START,
    ).run(max_transitions=TRANSITIONS)
