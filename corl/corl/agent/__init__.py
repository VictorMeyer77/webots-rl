import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from controller import Robot
from numpy.typing import NDArray

from corl.api.wrapper import Wrapper
from corl.schemas.learning import Observation
from corl.utils.config import Config

logger = logging.getLogger(__name__)

RETRY_MAX_ATTEMPTS = 5
RETRY_JITTER_BASE = 0.1


class Agent(ABC):
    """
    Abstract base class for a Webots reinforcement learning agent.

    Subclasses must implement :meth:`observe`, :meth:`policy`, and :meth:`act`,
    which together form the perception-decision-action cycle driven by
    :meth:`step`.

    Two execution modes are supported:

    - **Run mode** (``config["train"]`` is falsy): drives the robot with
      :meth:`run` until the Webots simulation stops.
    - **Training mode** (``config["train"]`` is truthy): drives a single
      training episode via :meth:`train`, exchanging observations, actions,
      and acknowledgments with the remote trainer through the
      :class:`~corl.api.wrapper.Wrapper` API.

    Attributes:
        robot: Webots ``Robot`` node being controlled.
        timestep: Simulation timestep in milliseconds.
        timestep_index: Cumulative count of simulation steps executed so far.
        action_repeat: Number of consecutive steps each action is held before
            a new perception-decision cycle is triggered.
        model: Optional loaded model used by :meth:`policy` (run mode).
        train_id: Training session identifier (training mode only).
        worker_id: Worker identifier within the training session (training mode only).
        api: :class:`~corl.api.wrapper.Wrapper` instance for API communication
            (training mode only).
    """

    robot: Robot
    timestep: int
    timestep_index: int
    action_repeat: int
    model: NDArray[np.float32] | None = None  # todo

    # Training attributes
    train_id: str | None = None
    worker_id: int | None = None
    api: Wrapper | None = None

    def __init__(
        self,
        robot: Robot,
        timestep: int,
        action_repeat: int,
        config: Config,
        model: NDArray[np.float32] | None = None,  # todo
    ):
        """
        Initialize the agent.

        In run mode (``config["train"]`` is falsy) only the robot and timestep
        attributes are set; all training attributes remain ``None``.

        In training mode (``config["train"]`` is truthy) the training
        attributes are populated and a :class:`~corl.api.wrapper.Wrapper`
        instance is created.

        Args:
            robot: Webots ``Robot`` node to control.
            timestep: Simulation timestep in milliseconds.
            action_repeat: Number of times to repeat each action before
                requesting a new one.
            config: Application configuration. Must contain ``"train"`` and,
                when ``"train"`` is truthy, ``"train_id"`` and ``"worker_id"``.
            model: Optional pre-loaded model used in run mode.
        """
        self.robot = robot
        self.timestep = timestep
        self.timestep_index = 0
        self.action_repeat = action_repeat
        self.model = model

        if config["train"]:
            self.train_id = config["train_id"]
            self.worker_id = config["worker_id"]
            self.api = Wrapper(config)
            logger.debug(
                f"Agent initialized for training with train_id={self.train_id} and worker_id={self.worker_id}"
            )

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

        Drives the Webots simulation step-by-step. On each iteration the
        current action is repeated up to ``action_repeat`` times before a
        fresh perception-decision cycle (:meth:`observe` → :meth:`policy` →
        :meth:`act`) is triggered. The loop exits when ``robot.step()``
        returns ``-1``, indicating that the Webots simulation has stopped.
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
            f"Webots simulation finish in {self.robot.getTime()} seconds with {self.timestep_index} timesteps."
        )

    # Training mode methods

    def train(self, max_timestep: int) -> None:
        """
        Run a single training episode up to ``max_timestep`` simulation steps.

        Fetches the current episode ID, then repeatedly sends observations to
        the remote trainer, retrieves the resulting action, and executes it
        until the simulation step budget is exhausted or the simulator stops.

        Args:
            max_timestep: Maximum number of simulation timesteps to run before
                terminating the training episode.

        Raises:
            RuntimeError: If the agent was not initialised in training mode
                (i.e. ``config["train"]`` was falsy).
            RuntimeError: If an observation fails to send or an action cannot
                be received after the maximum number of retries.
        """
        episode_id: int = self.api.get_episode_id(
            self.train_id, worker_id=self.worker_id
        )
        training_step: int = 0

        self.robot.step(self.timestep)

        while self.timestep_index < max_timestep:
            self._train_observe(episode_id, training_step)
            action = self._train_get_action(episode_id, training_step)
            self._train_execute_action(training_step, action)
            training_step += 1
            logger.debug(
                f"Completed training step {training_step} at timestep {self.robot.getTime()} for episode {episode_id}."
            )

    def _train_observe(self, episode_id: int, training_step: int) -> dict[str, Any]:
        """
        Collect an observation and send it to the remote trainer.

        Args:
            episode_id: Current episode identifier.
            training_step: Current step index within the episode.

        Returns:
            dict[str, Any]: The raw observation returned by :meth:`observe`.

        Raises:
            RuntimeError: If the observation cannot be sent to the API.
        """
        observation = self.observe()

        if not self.api.send_observation(
            self.train_id,
            self.worker_id,
            episode_id,
            training_step,
            Observation(data=observation),
        ):
            self._train_error(training_step, "Failed to send observation.")

        return observation

    def _train_get_action(self, episode_id: int, training_step: int) -> int:
        """
        Poll the API for the action corresponding to the current training step.

        Uses exponential back-off with jitter for retries. The first step is
        given additional leniency: while the worker status indicates it is still
        active, polling continues without consuming a retry slot.

        Args:
            episode_id: Current episode identifier.
            training_step: Current step index within the episode.

        Returns:
            int: The discrete action to execute.

        Raises:
            RuntimeError: If no action is received within ``RETRY_MAX_ATTEMPTS``
                retries.
        """
        retries = 0

        while retries < RETRY_MAX_ATTEMPTS:
            action = self.api.get_action(
                self.train_id, self.worker_id, episode_id, training_step
            )
            if action is not None:
                return action.action
            elif training_step == 0 and self.api.get_worker_status(
                self.train_id, self.worker_id
            ):
                logger.debug(
                    f"No action received for initial training step {training_step}. Retrying..."
                )
                time.sleep(np.random.uniform(0, 1))
            else:
                logger.debug(
                    f"No action received at training step {training_step}. Retrying (attempt {retries + 1}/{RETRY_MAX_ATTEMPTS})..."
                )
                max_delay = RETRY_JITTER_BASE * (2**retries)
                delay = np.random.uniform(0, max_delay)
                time.sleep(delay)
                retries += 1

        self._train_error(
            training_step, f"Failed to receive action after {retries} retries."
        )
        return -1  # Unreachable, but satisfies type checker

    def _train_execute_action(self, training_step: int, action: int) -> None:
        """
        Execute an action for ``action_repeat`` consecutive simulation steps.

        Calls :meth:`act` and advances the Webots simulation for each repeat.
        Exits early if the simulator signals termination (``robot.step`` returns
        ``-1``).

        Args:
            training_step: Current step index, used only for debug logging.
            action: Discrete action identifier to pass to :meth:`act`.
        """
        action_repeat_count = 0

        while action_repeat_count < self.action_repeat:
            self.act(action)

            if self.robot.step(self.timestep) != -1:
                action_repeat_count += 1
                self.timestep_index += 1
            else:
                logger.info(
                    f"Webots simulation finished during training at training step {training_step}."
                )
                break

        logger.debug(
            f"Executed action {action} for training step {training_step} with repeat count {self.action_repeat}. Timestep index: {self.timestep_index}"
        )

    def _train_error(self, training_step: int, error_message: str) -> None:
        """
        Mark the worker as failed and raise a ``RuntimeError``.

        Logs the error, notifies the API that this worker is no longer active,
        then raises an exception to abort the training episode.

        Args:
            training_step: The step at which the failure occurred.
            error_message: Human-readable description of the failure.

        Raises:
            RuntimeError: Always, with the step and worker context included.
        """
        logger.error(error_message)
        self.api.update_worker_status(self.train_id, self.worker_id, False)
        raise RuntimeError(
            f"Training step {training_step} failed: {error_message}. Worker {self.worker_id} marked as failed."
        )
