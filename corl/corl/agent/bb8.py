import logging
from typing import Any

from controller import Accelerometer, Gyro, Motor, Robot

from corl.agent.agent import Agent
from corl.model.model import Model
import numpy as np


MAX_VELOCITY = 8.72

logger = logging.getLogger(__name__)


class BB8(Agent):
    """
    Controller for the BB8 spherical robot in Webots.

    Extends :class:`~corl.agent.Agent` with BB8-specific hardware:
    body pitch and yaw motors, three accelerometers, and three gyroscopes.

    Accelerometers and gyroscopes are initialised automatically in ``__init__``.

    The discrete action space maps integer identifiers to
    ``(pitch_delta, yaw_delta)`` velocity pairs applied incrementally
    to the motors each call to :meth:`act`.

    Attributes:
        motors: Body pitch and yaw :class:`Motor` instances, in that order.
        accelerometers: List of three :class:`Accelerometer` instances
            (body, counterweight, head).
        gyros: List of three :class:`Gyro` instances
            (body, counterweight, head).
        actions: Mapping from action integer (``0``–``8``) to
            ``(pitch_delta, yaw_delta)`` velocity increment tuples.
    """

    motors: list[Motor] = []
    accelerometers: list[Accelerometer]
    gyroscopes: list[Gyro]
    actions: dict[int, tuple[float, float]]

    def __init__(
        self,
        robot: Robot,
        timestep: int,
        action_repeat: int,
        model: Model | None,
    ):
        """
        Initialize the BB8 agent, its motors, accelerometers, and gyroscopes.

        Calls the parent :meth:`~corl.agent.Agent.__init__`, initialises
        the pitch and yaw motors via :meth:`_init_motors`, enables all
        accelerometers and gyroscopes, and builds the action map.

        Args:
            robot: Webots ``Robot`` node to control.
            timestep: Simulation timestep in milliseconds.
            action_repeat: Number of consecutive simulation steps each
                selected action is held before a new one is requested.
            model: Optional pre-loaded model used by :meth:`~corl.agent.Agent.policy`.
        """
        super().__init__(
            robot=robot,
            timestep=timestep,
            action_repeat=action_repeat,
            model=model,
        )
        self._init_motors()
        self.init_accelerometers()
        self.init_gyros()

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

    def _init_motors(self) -> None:
        """
        Initialize the BB8 body pitch and yaw motors for velocity control.

        Configures both motors for continuous rotation by setting position to
        infinity and initial velocity to zero, enabling velocity control via
        setVelocity().
        """
        self.motors = [
            self.robot.getDevice("body pitch motor"),
            self.robot.getDevice("body yaw motor"),
        ]

        self.motors[0].setPosition(float("inf"))
        self.motors[1].setPosition(float("inf"))

        self.motors[0].setVelocity(0.0)
        self.motors[1].setVelocity(0.0)

        logger.debug("Motors initialized")

    def init_accelerometers(self) -> None:
        self.accelerometers = []
        names = [
            "body accelerometer",
            "counterweight accelerometer",
            "head accelerometer",
        ]
        for name in names:
            accelerometer = self.robot.getDevice(name)
            accelerometer.enable(self.timestep)
            self.accelerometers.append(accelerometer)
        logger.debug("Accelerometers initialized")

    def init_gyros(self) -> None:
        self.gyros = []
        names = ["body gyro", "counterweight gyro", "head gyro"]
        for name in names:
            gyro = self.robot.getDevice(name)
            gyro.enable(self.timestep)
            self.gyros.append(gyro)
        logger.debug("Gyros initialized")

    def observe(self) -> dict[str, Any]:

        observation = np.array(
            [
                [s.getValues() for s in self.accelerometers],
                [s.getValues() for s in self.gyros],
            ]
        ).flatten()

        return {"base": observation.tolist()}

    def act(self, action: int) -> None:

        if action not in self.actions:
            raise ValueError(
                f"Invalid action {action}. Must be one of {list(self.actions)}."
            )

        pitch_delta, yaw_delta = self.actions[action]

        new_pitch = np.clip(
            self.motors[0].getVelocity() + pitch_delta, -MAX_VELOCITY, MAX_VELOCITY
        )
        new_yaw = np.clip(
            self.motors[1].getVelocity() + yaw_delta, -MAX_VELOCITY, MAX_VELOCITY
        )

        self.motors[0].setVelocity(new_pitch)
        self.motors[1].setVelocity(new_yaw)

        logger.debug(
            f"BB8 action {action} executed: pitch velocity {self.motors[0].getVelocity()}, yaw velocity {self.motors[1].getVelocity()}"
        )
