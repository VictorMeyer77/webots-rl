
import numpy as np
from controller import Robot
from corl.agent.epuck import Epuck
from corl.model.genetic import ModelGenetic
from corl.model.model import Model
from corl.trainer.agent import TrainerAgent
from corl.utils.config import Config

from corl.utils.logger import setup_logging

TIME_STEP = 32
ACTION_REPEAT = 25
MAX_TIMESTEP = 1875


class EpuckGeneticController(Epuck):

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

    def policy(self, observation: dict) -> int:
            return self.model.predict(np.array([self.timestep_index // self.action_repeat], dtype=np.float32))


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
        model.load("/Users/victormeyer/Dev/Self/webots-rl/projects/.train/mlflow/4653918bab1c471daee903495c6f81e3/artifacts/model")


        epuck = EpuckGeneticController(
            robot=robot,
            timestep=TIME_STEP,
            action_repeat=ACTION_REPEAT,
            model=model,
        )
        epuck.run()

