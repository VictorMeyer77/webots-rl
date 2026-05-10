from typing import Any

import numpy as np
from controller import Robot

from corl.agent.bb8 import BB8
from corl.model.actor_critic_lite import ModelActorCriticLite
from corl.trainer.agent import TrainerAgent
from corl.utils.config import Config
from corl.utils.logger import setup_logging

TIME_STEP = 32  # Simulation timestep in milliseconds (15.625 Hz)
ACTION_REPEAT = 25  # Number of simulation timesteps to repeat each action
MAX_TIMESTEP = (
    1875  # Maximum number of simulation timesteps before reset (1875 * 32 ms = 60s)
)


class BB8PPOController(BB8):
    """
    BB-8 controller driven by a Proximal Policy Optimisation (PPO) policy.

    Reads the flat sensor observation vector and passes it to a pre-trained
    TFLite actor network to sample an action from the policy distribution.
    """

    def policy(self, observation: dict[str, Any]) -> int:
        """
        Sample an action from the actor's policy distribution for the current observation.

        The sensor vector is expanded to a batch of one before being passed
        to the TFLite actor network, which returns a sampled action index.

        Args:
            observation (dict[str, Any]): Must contain a ``"base"`` key with
                the current sensor vector as a float32 array of shape ``(INPUT_SHAPE,)``.

        Returns:
            int: Action index sampled from the actor's policy distribution.
        """
        observation_array = np.expand_dims(
            np.array(observation["base"], dtype=np.float32), axis=0
        )
        return int(self.model.predict(observation_array))


if __name__ == "__main__":
    config = Config()
    setup_logging(config)

    robot = Robot()

    if config.get("train_id") is not None:
        bb8 = BB8PPOController(
            robot=robot,
            timestep=TIME_STEP,
            action_repeat=ACTION_REPEAT,
            model=None,
        )
        TrainerAgent(bb8, config).run(MAX_TIMESTEP)

    else:
        model = ModelActorCriticLite()
        model.load(
            model_dir="/Users/victormeyer/Dev/Self/webots-rl/projects/.model/pit_escape/ppo"
        )

        bb8 = BB8PPOController(
            robot=robot,
            timestep=TIME_STEP,
            action_repeat=ACTION_REPEAT,
            model=model,
        )
        bb8.run()
