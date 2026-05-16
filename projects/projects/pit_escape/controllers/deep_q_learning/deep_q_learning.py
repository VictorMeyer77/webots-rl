from typing import Any

import numpy as np
from controller import Robot

from corl.agent.bb8 import BB8
from corl.model.discrete.deep_value_table_lite import ModelDeepValueTableLite
from corl.trainer.agent import TrainerAgent
from corl.utils.config import Config
from corl.utils.logger import setup_logging

TIME_STEP = 32  # Simulation timestep in milliseconds (15.625 Hz)
ACTION_REPEAT = 25  # Number of simulation timesteps to repeat each action
MAX_TIMESTEP = (
    1875  # Maximum number of simulation timesteps before reset (1875 * 32 ms = 60s)
)


class BB8DeepQLearningController(BB8):
    """
    BB-8 controller driven by a Deep Q-learning convolutional network policy.

    Captures grayscale camera frames and passes them to a pre-trained
    convolutional Q-network to select the greedy action.
    """

    def policy(self, observation: dict[str, Any]) -> list[float]:
        """
        Return the greedy action for the current observation.

        The camera frame is expanded to a batch of one before being passed
        to the convolutional Q-network.

        Args:
            observation (dict[str, Any]): Must contain a ``"base"`` key with
                the current frame as a float32 array.

        Returns:
            list[float]: Single-element list containing the greedy action
                index as a float (e.g. ``[3.0]``).
        """
        observation_array = np.expand_dims(
            np.array(observation["base"], dtype=np.float32), axis=0
        )
        return [float(self.model.predict(observation_array))]


if __name__ == "__main__":
    config = Config()
    setup_logging(config)

    robot = Robot()

    if config.get("train_id") is not None:
        bb8 = BB8DeepQLearningController(
            robot=robot,
            timestep=TIME_STEP,
            action_repeat=ACTION_REPEAT,
            model=None,
        )

        TrainerAgent(bb8, config).run(MAX_TIMESTEP)

    else:
        model = ModelDeepValueTableLite()
        model.load(
            model_dir="/Users/victormeyer/Dev/Self/webots-rl/projects/.model/pit_escape/deep_q_learning"
        )

        bb8 = BB8DeepQLearningController(
            robot=robot,
            timestep=TIME_STEP,
            action_repeat=ACTION_REPEAT,
            model=model,
        )
        bb8.run()
