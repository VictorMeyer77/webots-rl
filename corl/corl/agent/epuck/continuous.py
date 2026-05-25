import logging

import numpy as np

from corl.agent.epuck import MAX_VELOCITY, EpuckBase

logger = logging.getLogger(__name__)


class EpuckContinuous(EpuckBase):
    """
    Continuous-action e-puck controller.

    Accepts a two-element action ``[left_velocity, right_velocity]`` and sets
    the wheel motors to those absolute velocities directly, clipped to
    ``±MAX_VELOCITY``.  Designed for use with continuous-action algorithms
    such as SAC or DDPG where the policy outputs raw motor commands rather
    than discrete indices.
    """

    def act(self, action: list[float]) -> None:
        """
        Set absolute velocities for both wheel motors.

        Args:
            action: Two-element list ``[left_velocity, right_velocity]`` in
                normalised range ``[-1, 1]``. Values are scaled by
                ``MAX_VELOCITY`` and clipped before being applied.

        Raises:
            ValueError: If ``action`` does not have exactly two elements.
        """
        if len(action) != 2:
            raise ValueError(
                f"Continuous e-puck action must have 2 elements [left, right], "
                f"got {len(action)}: {action}."
            )

        left_vel = float(np.clip(action[0] * MAX_VELOCITY, -MAX_VELOCITY, MAX_VELOCITY))
        right_vel = float(
            np.clip(action[1] * MAX_VELOCITY, -MAX_VELOCITY, MAX_VELOCITY)
        )

        self.motors[0].setVelocity(left_vel)
        self.motors[1].setVelocity(right_vel)

        logger.debug(
            f"Epuck continuous action executed: left={left_vel:.3f}, right={right_vel:.3f}"
        )
