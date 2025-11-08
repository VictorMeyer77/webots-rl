from abc import ABC, abstractmethod

from brain.utils.logger import logger
from brain.utils.tcp_socket import TcpSocket
from controller import Robot


class BaseController(ABC):
    """
    Abstract base class for robot controllers.

    Attributes:
        robot (Robot): The robot instance to control.
        train (bool): Indicates if training mode is enabled.
        timestep (int): Simulation timestep in milliseconds.
        step_index (int): Current step index in the control loop.
        tcp_socket (TcpSocket | None): TCP socket for communication (used in training mode).

    This class defines the interface for observing, deciding actions, acting, and running the control loop.
    Subclasses must implement the abstract methods for specific robot behaviors.
    """

    robot: Robot
    train: bool
    timestep: int
    step_index: int
    tcp_socket: TcpSocket | None

    def __init__(self, robot: Robot, timestep: int = 64, train: bool = False):
        """
        Initialize the controller.

        Args:
            robot (Robot): The robot instance to control.
            timestep (int, optional): Simulation timestep in milliseconds. Defaults to 64.
            train (bool, optional): Whether to enable training mode. Defaults to False.
        """
        self.robot = robot
        self.timestep = timestep
        self.train = train
        self.step_index = 0
        if self.train:
            self.tcp_socket = TcpSocket(is_client=True)
        logger().info(f"Controller initialized in {'training' if self.train else 'production'} mode")

    @abstractmethod
    def observe(self) -> dict:
        """
        Read robot sensors.

        Returns:
                dict: A dictionary containing observation data.
        """
        raise NotImplementedError("Method observe() not implemented.")

    @abstractmethod
    def policy(self, observation: dict) -> int:
        """
        Decide on an action based on the observation.

        Args:
                observation (dict): The current robot sensors.

        Returns:
                int: The action to be taken.
        """
        raise NotImplementedError("Method policy() not implemented.")

    @abstractmethod
    def act(self, action: int):
        """
        Perform an action in the environment.

        Args:
                action (int): The action to be performed.
        """
        raise NotImplementedError("Method act() not implemented.")

    def step(self):
        """
        Execute one step of the control loop.
        """
        observations = self.observe()
        action = self.policy(observations)
        self.act(action)

    @abstractmethod
    def reset(self):
        """
        Reset controller for a new run.
        Use only for training.
        """
        self.step_index = 0
        logger().info("Controller has been reset")

    def run(self):
        """
        Run the controller loop.
        """
        while self.robot.step(self.timestep) != -1:
            self.step()
            self.step_index += 1
            logger().debug(f"Step index: {self.step_index}")
