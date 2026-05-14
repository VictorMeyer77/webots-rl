import logging

import numpy as np

from corl.agent.epuck import EpuckBase, MAX_VELOCITY

logger = logging.getLogger(__name__)


class EpuckDiscrete(EpuckBase):
    """
    Discrete-action e-puck controller.

    Maps integer action indices (``0``–``8``) to ``(left_delta, right_delta)``
    velocity increments applied to the wheel motors each call to :meth:`act`.

    Attributes:
        actions: Mapping from action index to ``(left_delta, right_delta)``
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
            0: (0.0, 0.0),  # Do nothing
            1: (0.1, 0.0),  # Turn left
            2: (0.0, 0.1),  # Turn right
            3: (0.1, 0.1),  # Move forward
            4: (-0.1, 0.0),  # Turn right backward
            5: (0.0, -0.1),  # Turn left backward
            6: (-0.1, -0.1),  # Move backward
            7: (0.1, -0.1),  # Quick right rotate
            8: (-0.1, 0.1),  # Quick left rotate
        }

    def act(self, action: list[float]) -> None:
        """
        Apply a velocity increment to both wheel motors.

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

        left_delta, right_delta = self.actions[action_index]

        new_left = np.clip(
            self.motors[0].getVelocity() + left_delta, -MAX_VELOCITY, MAX_VELOCITY
        )
        new_right = np.clip(
            self.motors[1].getVelocity() + right_delta, -MAX_VELOCITY, MAX_VELOCITY
        )

        self.motors[0].setVelocity(new_left)
        self.motors[1].setVelocity(new_right)

        logger.debug(
            f"Epuck discrete action {action_index} executed: "
            f"left={self.motors[0].getVelocity():.3f}, right={self.motors[1].getVelocity():.3f}"
        )
