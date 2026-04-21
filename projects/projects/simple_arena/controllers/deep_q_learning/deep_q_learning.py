from typing import Any

from controller import Robot
from corl.model.deep_value_table_lite import ModelDeepValueTableLite
from corl.agent.epuck import Epuck
from corl.model.model import Model
from corl.trainer.agent import TrainerAgent
from corl.utils.config import Config
from corl.utils.logger import setup_logging
import numpy as np

TIME_STEP = 32  # Simulation timestep in milliseconds (15.625 Hz)
ACTION_REPEAT = 25  # Number of simulation timesteps to repeat each action
MAX_TIMESTEP = (
    1875  # Maximum number of simulation timesteps before reset (1875 * 32 ms = 60s)
)
GRAYSCALE = True
NORMALIZE = True
IMAGE_SHAPE = (42, 42)


class EpuckDeepQLearningController(Epuck):
    """
    E-puck controller driven by a Deep Q-learning convolutional network policy.

    Captures grayscale camera frames and passes them to a pre-trained
    convolutional Q-network to select the greedy action.
    """

    def __init__(
        self,
        robot: Robot,
        timestep: int,
        action_repeat: int,
        model: Model | None = None,
    ):
        """
        Initialise the controller and activate the camera.

        Args:
            robot (Robot): Webots Robot instance.
            timestep (int): Simulation timestep in milliseconds.
            action_repeat (int): Number of simulation steps each action is held for.
            model (Model | None): Pre-trained convolutional Q-network model. Pass
                ``None`` during training; the trainer will supply actions externally.
        """
        super().__init__(
            robot=robot,
            timestep=timestep,
            action_repeat=action_repeat,
            model=model,
        )
        self.init_camera(
            image_shape=IMAGE_SHAPE, grayscale=GRAYSCALE, normalize=NORMALIZE
        )

    def policy(self, observation: dict[str, Any]) -> int:
        """
        Return the greedy action for the current observation.

        The camera frame is expanded to a batch of one before being passed
        to the convolutional Q-network.

        Args:
            observation (dict[str, Any]): Must contain a ``"camera"`` key with
                the current frame as a float32 array of shape ``IMAGE_SHAPE``.

        Returns:
            int: Greedy action index predicted by the Q-network.
        """
        observation_array = np.expand_dims(
            np.array(observation["camera"], dtype=np.float32), axis=0
        )
        return int(self.model.predict(observation_array))


if __name__ == "__main__":
    config = Config()
    setup_logging(config)

    robot = Robot()

    if config.get("train_id") is not None:
        epuck = EpuckDeepQLearningController(
            robot=robot,
            timestep=TIME_STEP,
            action_repeat=ACTION_REPEAT,
            model=None,
        )
        TrainerAgent(epuck, config).run(MAX_TIMESTEP)

    else:
        model = ModelDeepValueTableLite()
        model.load(
            model_dir="/Users/victormeyer/Dev/Self/webots-rl/projects/.model/deep_q_learning"
        )

        epuck = EpuckDeepQLearningController(
            robot=robot,
            timestep=TIME_STEP,
            action_repeat=ACTION_REPEAT,
            model=model,
        )

        epuck.run()
