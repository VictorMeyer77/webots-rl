import logging
from abc import ABC, abstractmethod
from typing import Any

from controller import Robot

from corl.model.model import Model

logger = logging.getLogger(__name__)


class Agent(ABC):
    """
    Abstract base class for a Webots reinforcement learning agent.

    Subclasses must implement :meth:`observe`, :meth:`policy`, and :meth:`act`,
    which together form the perception-decision-action cycle driven by
    :meth:`run`.

    Attributes:
        robot: Webots ``Robot`` node being controlled.
        timestep: Simulation timestep in milliseconds.
        timestep_index: Cumulative count of simulation steps executed so far.
        action_repeat: Number of consecutive steps each action is held before
            a new perception-decision cycle is triggered.
        model: Optional loaded model used by :meth:`policy`.
    """

    robot: Robot
    timestep: int
    timestep_index: int
    action_repeat: int
    model: Model | None

    def __init__(
        self, robot: Robot, timestep: int, action_repeat: int, model: Model | None
    ):
        """
        Initialize the agent.

        Args:
            robot: Webots ``Robot`` node to control.
            timestep: Simulation timestep in milliseconds.
            action_repeat: Number of times to repeat each action before
                requesting a new one.
            model: Optional pre-loaded model used by :meth:`policy`.
        """
        self.robot = robot
        self.timestep = timestep
        self.timestep_index = 0
        self.action_repeat = action_repeat
        self.model = model

    @abstractmethod
    def observe(self) -> dict[str, Any]:
        """
        Read robot sensors and build an observation payload.

        Subclasses must override this method to sample all relevant sensors
        and return their readings as a plain dictionary suitable for passing
        to :meth:`policy`.

        Returns:
            dict[str, Any]: Structured sensor data for policy consumption.
        """
        raise NotImplementedError("Method observe() not implemented.")

    @abstractmethod
    def policy(self, observation: dict[str, Any]) -> int:
        """
        Decide an action based on the current observation.

        Subclasses must override this method to implement the agent's
        decision logic, whether rule-based or model-based.

        Args:
            observation: Sensor-derived observation returned by :meth:`observe`.

        Returns:
            int: Discrete action identifier to be passed to :meth:`act`.
        """
        raise NotImplementedError("Method policy() not implemented.")

    @abstractmethod
    def act(self, action: int) -> None:
        """
        Execute the chosen action on the robot.

        Subclasses must override this method to translate the discrete action
        identifier into concrete motor or actuator commands.

        Args:
            action: Discrete action identifier returned by :meth:`policy`.
        """
        raise NotImplementedError("Method act() not implemented.")

    def run(self) -> None:
        """
        Continuous control loop until simulation termination.

        Performs one warm-up ``robot.step()`` call before entering the main
        loop to allow sensors to initialise. On each subsequent iteration a
        fresh perception-decision cycle (:meth:`observe` → :meth:`policy` →
        :meth:`act`) is triggered, and the resulting action is then held for
        up to ``action_repeat`` consecutive steps before the next cycle begins.
        ``timestep_index`` is incremented on every step, including repeated
        ones. The loop exits when ``robot.step()`` returns ``-1``, indicating
        that the Webots simulation has stopped.
        """

        current_action = None
        action_repeat_count = 0

        self.robot.step(self.timestep)

        while self.robot.step(self.timestep) != -1:
            if current_action is not None and action_repeat_count < self.action_repeat:
                action = current_action
                action_repeat_count += 1
            else:
                observation = self.observe()
                action = self.policy(observation)
                current_action = action
                action_repeat_count = 1

            self.act(action)
            self.timestep_index += 1

        logger.info(
            f"Webots simulation finished in {self.robot.getTime()} sim-seconds with {self.timestep_index} timesteps."
        )
