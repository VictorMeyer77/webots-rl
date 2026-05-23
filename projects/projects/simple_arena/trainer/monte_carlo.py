import logging

import numpy as np
from numpy.typing import NDArray

from corl.model.discrete.value_table import ModelValueTable
from corl.schemas.learning import Observation
from corl.schemas.tracker import StepKey
from corl.trainer.algorithm.discrete.monte_carlo import TrainerMonteCarlo
from corl.utils.config import Config
from corl.utils.logger import setup_logging

CHECKPOINT_FREQUENCY = 100_000  # Save a checkpoint every N epochs
TRANSITIONS = 500_000
ACTION_SIZE = 9  # Number of discrete actions available to the agent
OBSERVATION_CARDINALITY = (
    3  # Number of discrete bins per sensor (must equal len(bins) + 1)
)
OBSERVATION_SIZE = 8  # Number of distance sensors
BATCH_SIZE = 20  # Episodes collected per training batch
GAMMA = 0.99  # Discount factor: how much future rewards are valued
EPSILON = 1.0  # Initial exploration rate (fully random)
EPSILON_MIN = 0.01  # Minimum exploration rate after decay
EPSILON_DECAY = (
    0.99999  # Multiplicative decay applied each epoch (~floor reached at epoch 3200)
)
RETURNS_WINDOW = 100  # Max number of past returns kept per (state, action) pair


class SimpleArenaMonteCarlo(TrainerMonteCarlo):
    """
    Monte Carlo trainer for the simple arena task.

    Collects full episodes using an epsilon-greedy policy, then applies
    first-visit Monte Carlo updates to a discrete value table. Observations
    are distance-sensor readings discretised into 3 bins before lookup.
    """

    def parse_observations(
        self, observations: list[tuple[StepKey, Observation]]
    ) -> list[tuple[StepKey, NDArray[np.float32]]]:
        """
        Discretise raw distance-sensor readings for active workers.

        Filters out workers not currently tracked in the batch, then bins
        each sensor value into one of three categories using ``[70, 80]``
        as boundaries, matching ``OBSERVATION_CARDINALITY = 3``.

        Args:
            observations: Raw observations received from all workers in the
                current training step.

        Returns:
            List of ``(StepKey, binned_observation)`` pairs, one per active
            worker, where each binned observation is a float32 array of shape
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
            if step_key.worker_id in self.batch_worker_results
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

    SimpleArenaMonteCarlo(
        config=config,
        model=model,
        checkpoint_frequency=CHECKPOINT_FREQUENCY,
        #checkpoint_id="20260523_190936",
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        returns_window=RETURNS_WINDOW,
    ).run(max_transitions=TRANSITIONS)
