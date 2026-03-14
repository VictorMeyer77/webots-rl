import logging
import random
import time
from typing import Any, NoReturn

from corl.agent.agent import Agent
from corl.schemas.learning import Observation
from corl.trainer.wrapper import Wrapper
from corl.utils.config import Config

logger = logging.getLogger(__name__)

REFRESH_STATUS_INTERVAL = 5.0
REFRESH_EPISODE_INTERVAL = 0.5
RETRY_BASE_DELAY = 0.05
RETRY_MAX_DELAY = 2.0


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
        agent_request_timeout: Maximum seconds to wait for an action from the
            trainer before raising a :class:`RuntimeError`.
    """

    agent: Agent
    train_id: str
    worker_id: int
    api: Wrapper
    agent_request_timeout: float

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
        self.agent_request_timeout = config.get("agent_request_timeout")
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
            _ = self.send_observation(episode_id, training_step)
            action = self.get_action(episode_id, training_step)

            if self.execute_action(training_step, action):
                logger.debug(
                    f"Completed training step {training_step} at {self.agent.robot.getTime()} for episode {episode_id}."
                )
                training_step += 1
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

        Retries until an action is received or ``agent_request_timeout`` seconds
        have elapsed. Each retry sleeps for an exponentially increasing duration
        capped at ``RETRY_MAX_DELAY``, with ±50 % jitter to spread concurrent
        worker requests. Two periodic checks run during the wait:

        - Every ``REFRESH_STATUS_INTERVAL`` seconds: verify the worker is still
          active via :meth:`_check_worker_status`.
        - Every ``REFRESH_EPISODE_INTERVAL`` seconds: verify the episode ID has
          not changed via :meth:`_check_episode_id`.

        Args:
            episode_id: Current episode identifier.
            training_step: Current step index within the episode.

        Returns:
            The discrete action index to execute.

        Raises:
            RuntimeError: If no action is received within
                ``agent_request_timeout`` seconds.
        """

        deadline = time.monotonic() + self.agent_request_timeout
        last_status_check = time.monotonic()
        last_episode_check = time.monotonic()
        attempt = 0

        while time.monotonic() < deadline:
            action = self.api.get_action(
                self.train_id, self.worker_id, episode_id, training_step
            )

            if action is not None:
                return action.action

            now = time.monotonic()

            last_status_check = self._check_worker_status(
                training_step, now, last_status_check
            )

            last_episode_check = self._check_episode_id(
                episode_id, training_step, now, last_episode_check
            )

            delay = min(RETRY_BASE_DELAY * 2**attempt, RETRY_MAX_DELAY)
            time.sleep(delay * random.uniform(0.5, 1.0))
            attempt += 1

        self.propagate_error(
            training_step,
            f"Failed to receive action after {self.agent_request_timeout} seconds.",
        )

    def _check_worker_status(
        self, training_step: int, now: float, last_check: float
    ) -> float:
        """
        Periodically verify this worker is still active.

        If the API reports the worker as inactive, the simulator is advanced
        one tick so the environment can proceed with its shutdown sequence.

        Args:
            training_step: Current step index, used for logging.
            now: Current ``time.monotonic()`` timestamp.
            last_check: Timestamp of the previous status check.

        Returns:
            Updated ``last_check`` timestamp (``now``) if the interval has
            elapsed and the worker is still active, unchanged otherwise.

        Raises:
            RuntimeError: If the worker is found to be inactive.
        """
        if now - last_check > REFRESH_STATUS_INTERVAL:
            if not self.api.get_worker_status(self.train_id, self.worker_id):
                self.agent.robot.step(self.agent.timestep)
                self.propagate_error(
                    training_step,
                    f"Worker {self.worker_id} marked as inactive by API, agent should be shutting down by the environment.",
                )
            return now
        return last_check

    def _check_episode_id(
        self, episode_id: int, training_step: int, now: float, last_check: float
    ) -> float:
        """
        Periodically verify the current episode is still active.

        If the API returns a different episode ID, the simulator is advanced
        one tick so the environment can proceed with its shutdown sequence.

        Args:
            episode_id: Expected episode identifier.
            training_step: Current step index, used for logging.
            now: Current ``time.monotonic()`` timestamp.
            last_check: Timestamp of the previous episode check.

        Returns:
            Updated ``last_check`` timestamp (``now``) if the interval has
            elapsed and the episode is still active, unchanged otherwise.

        Raises:
            RuntimeError: If the episode ID returned by the API no longer
                matches ``episode_id``.
        """
        if now - last_check > REFRESH_EPISODE_INTERVAL:
            if self.api.get_episode_id(self.train_id, self.worker_id) != episode_id:
                self.agent.robot.step(self.agent.timestep)
                self.propagate_error(
                    training_step,
                    f"Episode {episode_id} marked as done by API, agent should be shutting down by the environment.",
                )
            return now
        return last_check

    def execute_action(self, training_step: int, action: int) -> bool:
        """
        Execute an action for ``action_repeat`` consecutive simulation steps.

        Calls :meth:`act` and advances the Webots simulation for each repeat.
        Exits early if the simulator signals termination (``robot.step`` returns
        ``-1``).

        Args:
            training_step: Current step index, used only for debug logging.
            action: Discrete action identifier to pass to :meth:`act`.

        Returns:
            ``True`` if all repeats completed normally, ``False`` if the
            simulator signalled termination before the repeats finished.
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
