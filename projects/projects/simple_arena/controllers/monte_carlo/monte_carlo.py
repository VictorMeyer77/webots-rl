from typing import Any

import numpy as np
from controller import Robot
from corl.agent.epuck import Epuck

from corl.model.model import Model
from corl.model.value_table import ModelValueTable
from corl.trainer.agent import TrainerAgent
from corl.utils.config import Config

from corl.utils.logger import setup_logging
from numpy.typing import NDArray

TIME_STEP = 32
ACTION_REPEAT = 25
MAX_TIMESTEP = 1875


class EpuckMonteCarloController(Epuck):
    def __init__(
        self,
        robot: Robot,
        timestep: int,
        action_repeat: int,
        model: Model | None = None,
    ):

        super().__init__(
            robot=robot,
            timestep=timestep,
            action_repeat=action_repeat,
            model=model,
        )

    def policy(self, observation: dict[str, Any]) -> int:

        if "distance_sensors" in observation:
            binned_observation = np.digitize(
                np.asarray(observation["distance_sensors"], dtype=np.float32),
                bins=[80, 70],
            )
            return self.model.predict(binned_observation.astype(np.float32))
        else:
            raise ValueError("Observation missing 'distance_sensors' key.")


if __name__ == "__main__":
    config = Config()
    setup_logging(config)

    robot = Robot()

    if config.get("train_id") is not None:
        epuck = EpuckMonteCarloController(
            robot=robot,
            timestep=TIME_STEP,
            action_repeat=ACTION_REPEAT,
            model=None,
        )
        TrainerAgent(epuck, config).run(MAX_TIMESTEP)
    else:
        model = ModelValueTable()
        model.load(
            "/Users/victormeyer/Dev/Self/webots-rl/projects/.train/mlflow/4d1c23ec2bd44cdf91a819d631d96924/artifacts/model"
        )

        epuck = EpuckMonteCarloController(
            robot=robot,
            timestep=TIME_STEP,
            action_repeat=ACTION_REPEAT,
            model=model,
        )
        epuck.run()
