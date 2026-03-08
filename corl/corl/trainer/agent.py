import logging
import time
from typing import Any, NoReturn

import numpy as np

from corl.agent import Agent
from corl.schemas.learning import Observation
from corl.trainer.wrapper import Wrapper
from corl.utils.config import Config

logger = logging.getLogger(__name__)

RETRY_MAX_ATTEMPTS = 8
RETRY_JITTER_BASE = 0.1


class TrainerAgent:
    """
    Bridge between a Webots :class:`~corl.agent.Agent` and a remote RL trainer.

    Drives the agent through a single training episode by exchanging
    observations and actions with the remote trainer over HTTP. Each decision
    cycle follows the pattern:

    1. **Observe** – collect sensor data via :meth:`send_observation`.
    2. **Act** – retrieve the chosen action via :meth:`get_action`.
    3. **Step** – execute the action for ``action_repeat`` ticks via
       :meth:`execute_action`.

    Retries are handled with exponential back-off (see :meth:`get_action`).
    Unrecoverable failures are reported to the API and raised as
    :class:`RuntimeError` via :meth:`propagate_error`.

    Attributes:
        agent: The :class:`~corl.agent.Agent` instance being trained.
        train_id: Unique identifier for the current training run.
        worker_id: Index of this worker within the training run.
        api: HTTP client used to communicate with the remote trainer.
    """

    agent: Agent
    train_id: str
    worker_id: int
    api: Wrapper

    def __init__(
        self,
        agent: Agent,
        config: Config,
    ):
        """
        Initialise the training wrapper.

        Args:
            agent: The :class:`~corl.agent.Agent` instance to drive during
                training.
            config: Application configuration. Must contain ``"train_id"``,
                ``"worker_id"``, ``"api_host"``, and ``"api_port"``.
        """
        self.agent = agent
        self.train_id = config.get("train_id")
        self.worker_id = config.get("worker_id")
        self.api = Wrapper(config)
        logger.debug(
            f"Agent initialized for training with train_id={self.train_id} and worker_id={self.worker_id}"
        )

    def run(self, max_timestep: int) -> None:
        """
        Run a single training episode up to ``max_timestep`` simulation steps.

        Fetches the current episode ID, performs one warm-up ``robot.step()``
        call to allow sensors to initialise (its return value is not checked),
        then repeatedly sends observations to the remote trainer, retrieves the
        resulting action, and executes it. The loop exits when any of the
        following conditions are met:

        - ``agent.timestep_index >= max_timestep`` (step budget exhausted), or
        - :meth:`execute_action` returns ``False`` (simulator stopped).

        Args:
            max_timestep: Maximum number of raw simulation ticks
                (``timestep_index``) before terminating the episode. Note that
                each decision cycle may advance the simulator by
                ``action_repeat`` ticks, so the number of training steps is
                approximately ``max_timestep // action_repeat``.
        """
        episode_id: int = self.api.get_episode_id(
            self.train_id, worker_id=self.worker_id
        )
        training_step: int = 0

        self.agent.robot.step(self.agent.timestep)

        while self.agent.timestep_index < max_timestep:
            self.send_observation(episode_id, training_step)
            action = self.get_action(episode_id, training_step)

            if self.execute_action(training_step, action):
                training_step += 1
                logger.debug(
                    f"Completed training step {training_step} at {self.agent.robot.getTime()} for episode {episode_id}."
                )
            else:
                logger.debug(
                    f"Simulator signaled termination at training step {training_step} for episode {episode_id}. Ending episode."
                )
                break

    def send_observation(self, episode_id: int, training_step: int) -> dict[str, Any]:
        """
        Collect an observation and send it to the remote trainer.

        Args:
            episode_id: Current episode identifier.
            training_step: Current step index within the episode.

        Returns:
            dict[str, Any]: The raw observation dict returned by
            :meth:`~corl.agent.Agent.observe`, passed through after being sent.

        Raises:
            RuntimeError: If the observation cannot be sent to the API.
        """
        observation = self.agent.observe()

        if not self.api.send_observation(
            self.train_id,
            self.worker_id,
            episode_id,
            training_step,
            Observation(data=observation),
        ):
            self.propagate_error(training_step, "Failed to send observation.")

        return observation

    def get_action(self, episode_id: int, training_step: int) -> int:
        """
        Poll the API for the action corresponding to the current training step.

        Uses exponential back-off with jitter for retries. Two special cases
        apply before the normal retry counter is incremented:

        - **Worker inactive** – if the API reports this worker as inactive,
          the simulator is advanced one tick and polling continues without
          consuming a retry slot.
        - **Initial step** (``training_step == 0``) – the first step is given
          unlimited leniency: polling retries indefinitely with a random
          sleep of up to 2 seconds and never increments the retry counter.

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
            elif not self.api.get_worker_status(self.train_id, self.worker_id):
                self.agent.robot.step(self.agent.timestep)
                logger.debug(
                    f"Worker {self.worker_id} marked as inactive by API at training step {training_step}. Should be shutting down by the environment."
                )
            elif training_step == 0:
                logger.debug(
                    "No action received for initial training step, retrying..."
                )
                time.sleep(np.random.uniform(0, 2))
            else:
                logger.debug(
                    f"No action received at training step {training_step}. Retrying (attempt {retries + 1}/{RETRY_MAX_ATTEMPTS})..."
                )
                max_delay = RETRY_JITTER_BASE * (2**retries)
                delay = np.random.uniform(0, max_delay)
                time.sleep(delay)
                retries += 1

        self.propagate_error(
            training_step, f"Failed to receive action after {retries} retries."
        )

    def execute_action(self, training_step: int, action: int) -> bool:
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

        while action_repeat_count < self.agent.action_repeat:
            self.agent.act(action)

            if self.agent.robot.step(self.agent.timestep) != -1:
                action_repeat_count += 1
                self.agent.timestep_index += 1
            else:
                return False

        logger.debug(
            f"Executed action {action} for training step {training_step} with repeat count {self.agent.action_repeat}."
        )
        return True

    def propagate_error(self, training_step: int, error_message: str) -> NoReturn:
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
