from typing import Any

import numpy as np
from controller import Robot

from corl.agent.bb8.continuous import BB8Continuous
from corl.model.actor_critic_lite import ModelActorCriticLite
from corl.trainer.agent import TrainerAgent
from corl.utils.config import Config
from corl.utils.logger import setup_logging

TIME_STEP = 32  # Simulation timestep in milliseconds (15.625 Hz)
ACTION_REPEAT = 25  # Number of simulation timesteps to repeat each action
MAX_TIMESTEP = (
    1875  # Maximum number of simulation timesteps before reset (1875 * 32 ms = 60s)
)
ACTION_DIM = 2  # [pitch_velocity, yaw_velocity]


class BB8SACController(BB8Continuous):
    """
    BB-8 controller driven by a Soft Actor-Critic (SAC) policy.

    Reads the flat sensor observation vector (accelerometers + gyroscopes)
    and passes it to a pre-trained TFLite actor network to produce continuous
    motor velocities ``[pitch_velocity, yaw_velocity]`` in the normalised
    range ``[-1, 1]``.
    """

    def policy(self, observation: dict[str, Any]) -> list[float]:
        """
        Predict continuous motor velocities for the current observation.

        The sensor vector is expanded to a batch of one before being passed
        to the TFLite actor network, which returns the mean action
        ``[pitch_velocity, yaw_velocity]`` in ``[-1, 1]``.

        Args:
            observation (dict[str, Any]): Must contain a ``"base"`` key with
                the current sensor vector as a flat float32 array (18 values:
                3 accelerometers × 3 axes + 3 gyroscopes × 3 axes).

        Returns:
            list[float]: Two-element list ``[pitch_velocity, yaw_velocity]``
                in the normalised range ``[-1, 1]``.
        """
        observation_array = np.expand_dims(
            np.array(observation["base"], dtype=np.float32), axis=0
        )
        return self.model.predict(observation_array).flatten().tolist()


if __name__ == "__main__":
    config = Config()
    setup_logging(config)

    robot = Robot()

    if config.get("train_id") is not None:
        bb8 = BB8SACController(
            robot=robot,
            timestep=TIME_STEP,
            action_repeat=ACTION_REPEAT,
            model=None,
        )
        TrainerAgent(bb8, config).run(MAX_TIMESTEP)

    else:
        model = ModelActorCriticLite()
        model.load(
            model_dir="/Users/victormeyer/Dev/Self/webots-rl/projects/.model/pit_escape/sac"
        )

        bb8 = BB8SACController(
            robot=robot,
            timestep=TIME_STEP,
            action_repeat=ACTION_REPEAT,
            model=model,
        )

        bb8.run()
