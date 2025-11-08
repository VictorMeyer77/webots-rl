import json
from abc import ABC, abstractmethod

from brain.utils.logger import logger
from brain.utils.tcp_socket import TcpSocket
from controller import Supervisor


class EnvironmentState:
    """
    Represents the current state of the environment during agent execution.

    This class tracks the environment's execution state including termination status,
    success status, and the current step count. It provides JSON serialization
    for state persistence and logging.

    Attributes:
        is_terminated (bool): Whether the environment has reached a terminal state
        is_success (bool): Whether the environment task completed successfully
        step_index (int): Current step number in the environment execution
    """

    is_terminated: bool
    is_success: bool
    step_index: int

    def __init__(self):
        """
        Initialize the environment state with default values.

        Sets is_terminated and is_success to False, and step_index to 0.
        """
        self.is_terminated = False
        self.is_success = False
        self.step_index = 0

    def to_json(self) -> str:
        """
        Serialize the environment state to a JSON string.

        Converts all instance attributes to a JSON-formatted string representation.

        Returns:
            str: JSON string containing all state attributes
        """
        return json.dumps(self.__dict__)


class Environment(ABC):
    """
    Abstract base class for simulation environments.

    This class defines the interface and common attributes for environments
    used in reinforcement learning or control tasks. Subclasses must implement
    the required abstract methods to define environment-specific behavior.

    Attributes:
        supervisor (Supervisor): The Webots Supervisor instance managing the simulation.
        train (bool): Indicates if the environment is in training mode.
        tcp_socket (TcpSocket | None): TCP socket for communication during training, if enabled.
        step_index (int): Current step number in the environment execution.
        max_step (int): Maximum number of steps allowed in an episode.
    """

    supervisor: Supervisor
    train: bool = False
    tcp_socket: TcpSocket | None
    step_index: int
    max_step: int

    def __init__(self, supervisor: Supervisor, max_step: int, train: bool = False):
        """
        Initialize the environment with a Supervisor instance.

        Args:
            supervisor (Supervisor): The Webots Supervisor instance managing the simulation.
            max_step (int): Maximum number of steps allowed in an episode.
            train (bool): Whether to enable training mode. Defaults to False.
        """
        self.supervisor = supervisor
        self.train = train
        self.step_index = 0
        self.max_step = max_step
        if self.train:
            self.tcp_socket = TcpSocket(is_client=False)
        logger().debug(
            f"Environment initialized in {'training' if self.train else 'production'} mode. Max steps: {self.max_step}"
        )

    @abstractmethod
    def step(self) -> tuple[EnvironmentState, float]:
        """
        Advance the simulation by one timestep.

        Returns
            tuple[EnvironmentState, float]: Tuple with new environment state and reward.

        """
        raise NotImplementedError("Method step() not implemented.")

    @abstractmethod
    def state(self) -> EnvironmentState:
        raise NotImplementedError("Method state() not implemented.")

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.supervisor.simulationReset()
        self.step_index = 0
        logger().info("Environment reset.")

    def quit(self):
        """
        Quit the simulation.
        """
        self.supervisor.simulationQuit(0)
        logger().info("Environment terminated successfully.")

    @abstractmethod
    def run(self) -> float:
        """
        Execute simulation and return total reward.

        Returns:
            float: The cumulated reward.
        """
        raise NotImplementedError("Method run() not implemented.")
