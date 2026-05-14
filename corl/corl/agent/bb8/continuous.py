import logging

import numpy as np

from corl.agent.bb8 import BB8Base, MAX_VELOCITY

logger = logging.getLogger(__name__)


class BB8Continuous(BB8Base):
    """
    Continuous-action BB8 controller.

    Accepts a two-element action ``[pitch_velocity, yaw_velocity]`` and sets
    the motors to those absolute velocities directly, clipped to
    ``±MAX_VELOCITY``.  Designed for use with continuous-action algorithms
    such as SAC or DDPG.
    """

    def act(self, action: list[float]) -> None:
        """
        Set absolute velocities for pitch and yaw motors.

        Args:
            action: Two-element list ``[pitch_velocity, yaw_velocity]`` in
                normalised range ``[-1, 1]``. Values are scaled by
                ``MAX_VELOCITY`` and clipped before being applied.

        Raises:
            ValueError: If ``action`` does not have exactly two elements.
        """
        if len(action) != 2:
            raise ValueError(
                f"Continuous BB8 action must have 2 elements [pitch, yaw], "
                f"got {len(action)}: {action}."
            )

        pitch_vel = float(
            np.clip(action[0] * MAX_VELOCITY, -MAX_VELOCITY, MAX_VELOCITY)
        )
        yaw_vel = float(np.clip(action[1] * MAX_VELOCITY, -MAX_VELOCITY, MAX_VELOCITY))

        self.motors[0].setVelocity(pitch_vel)
        self.motors[1].setVelocity(yaw_vel)

        logger.debug(
            f"BB8 continuous action executed: pitch={pitch_vel:.3f}, yaw={yaw_vel:.3f}"
        )
