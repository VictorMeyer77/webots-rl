import logging
from typing import Any

import numpy as np
from controller import Accelerometer, Gyro, Motor, Robot

from corl.agent.agent import Agent
from corl.model.model import Model

MAX_VELOCITY = 8.72

logger = logging.getLogger(__name__)


class BB8Base(Agent):
    """
    Hardware base class for the BB8 spherical robot in Webots.

    Manages body pitch and yaw motors, three accelerometers, and three
    gyroscopes.  Does **not** define an action space — subclasses implement
    :meth:`act` for their respective space (discrete or continuous).

    Sensors are initialised automatically in ``__init__``.

    Attributes:
        motors: Body pitch and yaw :class:`Motor` instances, in that order.
        accelerometers: List of three :class:`Accelerometer` instances
            (body, counterweight, head).
        gyroscopes: List of three :class:`Gyro` instances
            (body, counterweight, head).
    """

    motors: list[Motor] = []
    accelerometers: list[Accelerometer]
    gyroscopes: list[Gyro]

    def __init__(
        self,
        robot: Robot,
        timestep: int,
        action_repeat: int,
        model: Model | None,
    ):
        """
        Initialize the BB8 base agent, its motors, accelerometers, and gyroscopes.

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

    def _init_motors(self) -> None:
        """
        Initialize the BB8 body pitch and yaw motors for velocity control.

        Configures both motors for continuous rotation by setting position to
        infinity and initial velocity to zero.
        """
        self.motors = [
            self.robot.getDevice("body pitch motor"),
            self.robot.getDevice("body yaw motor"),
        ]
        self.motors[0].setPosition(float("inf"))
        self.motors[1].setPosition(float("inf"))
        self.motors[0].setVelocity(0.0)
        self.motors[1].setVelocity(0.0)
        logger.debug("BB8 motors initialized")

    def init_accelerometers(self) -> None:
        """Initialize and enable the three BB8 accelerometers."""
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
        logger.debug("BB8 accelerometers initialized")

    def init_gyros(self) -> None:
        """
        Initialize and enable the three BB8 gyroscopes.

        Gyroscopes initialized: ``"body gyro"``, ``"counterweight gyro"``,
        ``"head gyro"``. Each is enabled at the simulation ``timestep``
        set during construction.
        """
        self.gyros = []
        names = ["body gyro", "counterweight gyro", "head gyro"]
        for name in names:
            gyro = self.robot.getDevice(name)
            gyro.enable(self.timestep)
            self.gyros.append(gyro)
        logger.debug("BB8 gyros initialized")

    def observe(self) -> dict[str, Any]:
        """
        Read accelerometers and gyroscopes and return the observation payload.

        Returns:
            dict with key ``"base"`` containing a flat ``list[float]`` of all
            accelerometer and gyroscope values (18 values total).
        """
        observation = np.array(
            [
                [s.getValues() for s in self.accelerometers],
                [s.getValues() for s in self.gyros],
            ]
        ).flatten()
        return {"base": observation.tolist()}


# Backward-compatibility alias — existing ``from corl.agent.bb8 import BB8``
# imports continue to resolve to BB8Discrete.
from corl.agent.bb8.discrete import BB8Discrete  # noqa: E402

BB8 = BB8Discrete
