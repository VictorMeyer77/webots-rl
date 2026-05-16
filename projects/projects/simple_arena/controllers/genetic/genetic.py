from typing import Any

import numpy as np
from controller import Robot

from corl.agent.epuck.discrete import EpuckDiscrete
from corl.model.discrete.genetic import ModelGenetic
from corl.trainer.agent import TrainerAgent
from corl.utils.config import Config
from corl.utils.logger import setup_logging

TIME_STEP = 32  # Simulation timestep in milliseconds (15.625 Hz)
ACTION_REPEAT = 25  # Number of simulation timesteps to repeat each action
MAX_TIMESTEP = 1875  # Maximum steps per episode (1875 * 32 ms = 60 seconds)


class EpuckGeneticController(EpuckDiscrete):
    """
    E-puck controller driven by a pre-evolved genetic genome.

    Implements an open-loop policy: the action at each step is read directly
    from the genome by index, ignoring sensor observations entirely. The genome
    is a fixed sequence of actions evolved offline and replayed during inference.
    """

    def policy(self, _observation: dict[str, Any]) -> list[float]:
        """
        Return the genome action for the current step index.

        The observation is intentionally ignored — this is an open-loop
        controller that replays a pre-evolved action sequence.

        Args:
            _observation: Sensor data from the environment (unused).

        Returns:
            list[float]: Single-element list with the action index read from
                the genome at position ``timestep_index // action_repeat`` as a
                float (e.g. ``[3.0]``).
        """
        return [
            float(
                self.model.predict(
                    np.array(
                        [self.timestep_index // self.action_repeat], dtype=np.float32
                    )
                )
            )
        ]


if __name__ == "__main__":
    config = Config()
    setup_logging(config)

    robot = Robot()

    if config.get("train_id") is not None:
        epuck = EpuckGeneticController(
            robot=robot,
            timestep=TIME_STEP,
            action_repeat=ACTION_REPEAT,
            model=None,
        )
        TrainerAgent(epuck, config).run(MAX_TIMESTEP)
    else:
        model = ModelGenetic()
        model.load(
            "/Users/victormeyer/Dev/Self/webots-rl/projects/.train/mlflow/079a04935fed4c06aaedf2293962aefb/artifacts/model"
        )

        epuck = EpuckGeneticController(
            robot=robot,
            timestep=TIME_STEP,
            action_repeat=ACTION_REPEAT,
            model=model,
        )
        epuck.run()
