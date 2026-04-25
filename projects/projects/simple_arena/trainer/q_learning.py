import logging

import numpy as np
from numpy.typing import NDArray

from corl.model.value_table import ModelValueTable
from corl.schemas.learning import Observation
from corl.schemas.tracker import StepKey
from corl.trainer.algorithm.q_learning import TrainerQLearning
from corl.utils.config import Config
from corl.utils.logger import setup_logging

TRANSITIONS = 1_000_000  # Total number of training epochs
MODEL_CHECKPOINT_FREQUENCY = 250_000  # Save a checkpoint every N epochs
ACTION_SIZE = 9  # Number of discrete actions available to the agent
OBSERVATION_CARDINALITY = (
    3  # Number of discrete bins per sensor (must equal len(bins) + 1)
)
OBSERVATION_SIZE = 8  # Number of distance sensors
ALPHA = 0.05  # Learning rate: how much new estimates overwrite old ones
GAMMA = 0.97  # Discount factor: how much future rewards are valued
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999995


class SimpleArenaQLearning(TrainerQLearning):
    """
    Q-learning trainer for the simple arena task.

    Applies online TD updates to a discrete Q-table after each step using
    an epsilon-greedy policy. Observations are distance-sensor readings
    discretised into 3 bins before lookup.
    """

    def parse_observations(
        self, observations: list[tuple[StepKey, Observation]]
    ) -> list[tuple[StepKey, NDArray[np.float32]]]:
        """
        Discretise raw distance-sensor readings for all workers.

        Bins each sensor value into one of three categories using ``[70, 80]``
        as boundaries, matching ``OBSERVATION_CARDINALITY = 3``.

        Args:
            observations: Raw observations received from all workers in the
                current training step.

        Returns:
            List of ``(StepKey, binned_observation)`` pairs for every worker,
            where each binned observation is a float32 array of shape
            ``(OBSERVATION_SIZE,)`` with values in ``{0, 1, 2}``.
        """
        return [
            (
                step_key,
                np.digitize(
                    np.asarray(observation.data["distance_sensors"], dtype=np.float32),
                    bins=[70, 80],
                ).astype(np.float32),
            )
            for step_key, observation in observations
        ]


if __name__ == "__main__":
    config = Config()
    setup_logging(config)
    logger = logging.getLogger(__name__)

    # model = ModelValueTable(
    #    model_dir="/Users/victormeyer/Dev/Self/webots-rl/projects/.train/mlflow/9e706c4ed2e9417eb85a29986a0b76a6/artifacts/model"
    # )

    model = ModelValueTable(
        observation_cardinality=OBSERVATION_CARDINALITY,
        observation_size=OBSERVATION_SIZE,
        action_size=ACTION_SIZE,
    )

    SimpleArenaQLearning(
        config=config,
        model=model,
        model_checkpoint_frequency=MODEL_CHECKPOINT_FREQUENCY,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
    ).run(epochs=TRANSITIONS)
