"""
Base controller abstractions for Webots reinforcement learning.

This module defines `BaseController`, an abstract interface for robot controllers
that perform a perception-decision-action loop each simulation timestep.

Responsibilities:
- Observing sensor data (`observe()`).
- Selecting actions via a policy (`policy()`).
- Executing actions on the robot (`act()`).
- Running a continuous control loop (`run()`).
- Optional training routine (`train()`).

Subclasses must implement all abstract methods to provide concrete behavior.
"""

from abc import ABC, abstractmethod

from brain.model import Model
from brain.utils.logger import logger
from brain.utils.queue import Queue
from controller import Robot


class BaseController(ABC):
    """
    Abstract base class for Webots robot controllers.

    Attributes:
        robot (Robot): Controlled robot instance.
        timestep (int): Simulation step duration in milliseconds.
        queue (Queue | None): Message queue for supervisor/robot communication.
    """

    robot: Robot
    timestep: int
    queue: Queue | None
    model: Model

    def __init__(self, robot: Robot, timestep: int = 64):
        """
        Initialize the controller.

        Args:
            robot (Robot): Robot instance to control.
            timestep (int): Simulation timestep in milliseconds (default: 64).
        """
        self.robot = robot
        self.timestep = timestep
        self.queue = Queue(timestep, robot.getDevice("emitter"), robot.getDevice("receiver"))
        self.model = None

    @abstractmethod
    def observe(self) -> dict:
        """
        Read robot sensors and build an observation payload.

        Returns:
            dict: Structured sensor data for policy consumption.
        """
        raise NotImplementedError("Method observe() not implemented.")

    @abstractmethod
    def policy(self, observation: dict) -> int:
        """
        Decide an action based on the current observation.

        Args:
            observation (dict): Sensor-derived observation.

        Returns:
            int: Discrete action identifier.
        """
        raise NotImplementedError("Method policy() not implemented.")

    @abstractmethod
    def act(self, action: int) -> None:
        """
        Execute the chosen action on the robot.

        Args:
            action (int): Action identifier.
        """
        raise NotImplementedError("Method act() not implemented.")

    def step(self) -> None:
        """
        Perform one perception-decision-action cycle.
        """
        observation = self.observe()
        action = self.policy(observation)
        self.act(action)

    def run(self) -> None:
        """
        Continuous control loop until simulation termination.
        """
        step_index = 0
        while self.robot.step(self.timestep) != -1:
            self.step()
            step_index += 1
            logger().debug(f"Step index: {step_index}")

    @abstractmethod
    def train(self) -> None:
        """
        Execute a training routine (e.g., data collection or policy update).
        """
        raise NotImplementedError("Method train() not implemented.")

    def set_model(self, model: Model) -> None:
        self.model = model
