from typing import Any

import numpy as np
from controller import Robot

from corl.agent.epuck.continuous import EpuckContinuous
from corl.model.actor_critic_lite import ModelActorCriticLite
from corl.model.model import Model
from corl.trainer.agent import TrainerAgent
from corl.utils.config import Config
from corl.utils.logger import setup_logging

TIME_STEP = 32  # Simulation timestep in milliseconds (15.625 Hz)
ACTION_REPEAT = 25  # Number of simulation timesteps to repeat each action
MAX_TIMESTEP = (
    1875  # Maximum number of simulation timesteps before reset (1875 * 32 ms = 60s)
)
ACTION_DIM = 2  # [left_velocity, right_velocity]
GRAYSCALE = True
NORMALIZE = True
IMAGE_SHAPE = (42, 42)


class EpuckSACController(EpuckContinuous):
    """
    E-puck controller driven by a Soft Actor-Critic (SAC) policy.

    Captures grayscale camera frames and passes them to a pre-trained
    TFLite actor network to produce continuous wheel velocities
    ``[left_velocity, right_velocity]`` in the normalised range ``[-1, 1]``.
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
            action_repeat (int): Number of simulation steps each chosen action
                is held for.
            model (Model | None): Pre-trained SAC actor TFLite model. Pass
                ``None`` during training; the trainer will supply actions
                externally.
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

    def policy(self, observation: dict[str, Any]) -> list[float]:
        """
        Predict continuous wheel velocities for the current observation.

        The camera frame is flattened and expanded to a batch of one before
        being passed to the TFLite actor network, which returns the mean
        action ``[left_velocity, right_velocity]`` in ``[-1, 1]``.

        Args:
            observation: Must contain a ``"camera"`` key with the current
                frame as a float32 array of shape ``IMAGE_SHAPE``.

        Returns:
            list[float]: Two-element list ``[left_velocity, right_velocity]``
                in the normalised range ``[-1, 1]``.
        """
        observation_array = np.expand_dims(
            np.array(observation["camera"], dtype=np.float32).flatten(), axis=0
        )
        return self.model.predict(observation_array).flatten().tolist()


if __name__ == "__main__":
    config = Config()
    setup_logging(config)

    robot = Robot()

    if config.get("train_id") is not None:
        epuck = EpuckSACController(
            robot=robot,
            timestep=TIME_STEP,
            action_repeat=ACTION_REPEAT,
            model=None,
        )
        TrainerAgent(epuck, config).run(MAX_TIMESTEP)

    else:
        model = ModelActorCriticLite()
        model.load(
            model_dir="/Users/victormeyer/Dev/Self/webots-rl/projects/.model/simple_arena/sac"
        )

        epuck = EpuckSACController(
            robot=robot,
            timestep=TIME_STEP,
            action_repeat=ACTION_REPEAT,
            model=model,
        )

        epuck.run()
