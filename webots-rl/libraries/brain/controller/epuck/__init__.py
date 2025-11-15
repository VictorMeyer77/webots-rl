"""
This module provides base classes for controlling an e-puck robot in Webots.

Classes:
    - BaseEpuck: Initializes distance sensors and motors, provides observation and reset methods.
    - EpuckTurner: Implements discrete movement actions (forward, left, right, backward) for the e-puck robot.

These classes are designed for extensibility and integration with training or simulation environments.
"""

from brain.controller import BaseController
from brain.utils.logger import logger
from controller import DistanceSensor, Motor, Robot


class BaseEpuck(BaseController):
    """
    Base controller for the e-puck robot in Webots.

    Attributes:
        max_speed (float): Maximum speed for the robot's motors.
        distance_sensors (list[DistanceSensor]): List of distance sensor devices.
        motors (list[Motor]): List containing left and right wheel motor devices.
    """

    max_speed: float
    distance_sensors: list[DistanceSensor]
    motors: list[Motor]

    def __init__(self, robot: Robot, timestep: int, max_speed: float):
        """
        Initialize the BaseEpuck controller.

        Args:
            robot (Robot): The Webots robot instance.
            timestep (int): Simulation timestep in milliseconds.
            max_speed (float): Maximum speed for the robot's motors.
        """
        super().__init__(robot=robot, timestep=timestep)
        self.max_speed = max_speed
        self._init_distance_sensors()
        self._init_motors()

    def _init_distance_sensors(self) -> None:
        """
        Initialize and enable the e-puck's distance sensors.

        Sets up eight distance sensors by retrieving them from the robot and enabling them
        with the simulation timestep.
        """

        self.distance_sensors = []

        names = ["ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7"]

        for i in range(8):
            self.distance_sensors.append(self.robot.getDevice(names[i]))
            self.distance_sensors[i].enable(self.timestep)
        logger().debug("Epuck distance sensors initialized")

    def _init_motors(self) -> None:
        """
        Initialize the e-puck's wheel motors for velocity control.

        Retrieves the left and right wheel motors, sets their position to infinity for continuous rotation,
        and initializes their velocity to zero.
        """
        self.motors = [self.robot.getDevice("left wheel motor"), self.robot.getDevice("right wheel motor")]
        self.motors[0].setPosition(float("inf"))
        self.motors[1].setPosition(float("inf"))
        self.motors[0].setVelocity(0.0)
        self.motors[1].setVelocity(0.0)
        logger().debug("Epuck motors initialized")

    def observe(self) -> dict:
        """
        Collect current readings from all distance sensors.

        Returns:
            dict: Dictionary containing a list of distance sensor values under the key 'distance_sensors'.
        """
        observations = {"distance_sensors": [s.getValue() for s in self.distance_sensors]}
        logger().debug(f"Epuck distance sensor readings: {observations['distance_sensors']}")
        return observations


class EpuckTurner(BaseEpuck):
    """
    Controller for the e-puck robot that supports discrete turning actions.
    """

    def __init__(self, robot: Robot, timestep: int, max_speed: float):

        super().__init__(robot=robot, timestep=timestep, max_speed=max_speed)

        self.actions = {
            0: [0.5 * self.max_speed, 0.5 * self.max_speed],  # forward
            1: [-0.5 * self.max_speed, 0.5 * self.max_speed],  # left
            2: [0.5 * self.max_speed, -0.5 * self.max_speed],  # right
            3: [-0.5 * self.max_speed, -0.5 * self.max_speed],  # backward
        }

    def act(self, action: int) -> None:
        """
        Set the wheel velocities based on the given action.

        Actions:
            0: Move forward
            1: Turn left
            2: Turn right
            3: Move backward

        Args:
            action (int): Discrete action (0-3).

        Raises:
            ValueError: If action is not between 0 and 3.
        """

        if action not in self.actions:
            raise ValueError("Action must be an integer between 0 and 3.")

        logger().debug(
            f"Epuck motor velocities updated with left_speed = {self.actions[action][0]} "
            f"and right_speed = {self.actions[action][1]}"
        )

        self.motors[0].setVelocity(self.actions[action][0])
        self.motors[1].setVelocity(self.actions[action][1])
