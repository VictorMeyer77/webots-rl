import logging

import numpy as np

from corl.agent.bb8 import BB8Base, MAX_VELOCITY

logger = logging.getLogger(__name__)


class BB8Discrete(BB8Base):
    """
    Discrete-action BB8 controller.

    Maps integer action indices (``0``–``8``) to ``(pitch_delta, yaw_delta)``
    velocity increments applied to the motors each call to :meth:`act`.

    Attributes:
        actions: Mapping from action index to ``(pitch_delta, yaw_delta)``
            velocity increment tuples.
    """

    actions: dict[int, tuple[float, float]]

    def __init__(self, robot, timestep: int, action_repeat: int, model):
        super().__init__(
            robot=robot,
            timestep=timestep,
            action_repeat=action_repeat,
            model=model,
        )
        self.actions = {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (0.0, 1.0),
            3: (1.0, 1.0),
            4: (-1.0, 0.0),
            5: (0.0, -1.0),
            6: (-1.0, -1.0),
            7: (1.0, -1.0),
            8: (-1.0, 1.0),
        }

    def act(self, action: list[float]) -> None:
        """
        Apply a velocity increment to both motors.

        Args:
            action: Single-element list with the action index as a float
                (e.g. ``[3.0]``). Must be a key in :attr:`actions` (``0``–``8``).

        Raises:
            ValueError: If the action index is not a valid key.
        """
        action_index = int(action[0])
        if action_index not in self.actions:
            raise ValueError(
                f"Invalid action {action_index}. Must be one of {list(self.actions)}."
            )

        pitch_delta, yaw_delta = self.actions[action_index]

        new_pitch = np.clip(
            self.motors[0].getVelocity() + pitch_delta, -MAX_VELOCITY, MAX_VELOCITY
        )
        new_yaw = np.clip(
            self.motors[1].getVelocity() + yaw_delta, -MAX_VELOCITY, MAX_VELOCITY
        )

        self.motors[0].setVelocity(new_pitch)
        self.motors[1].setVelocity(new_yaw)

        logger.debug(
            f"BB8 discrete action {action_index} executed: "
            f"pitch={self.motors[0].getVelocity():.3f}, yaw={self.motors[1].getVelocity():.3f}"
        )
