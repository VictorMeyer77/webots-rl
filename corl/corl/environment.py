import logging
import time
from abc import ABC, abstractmethod

from controller import Supervisor

from corl.api.wrapper import Wrapper
from corl.schemas.learning import Environment as EnvironmentSchema
from corl.utils.config import Config

logger = logging.getLogger(__name__)


class Environment(ABC):
    """
    Abstract base class for a Webots reinforcement learning environment.

    Subclasses must implement :meth:`step`, which advances the simulation by
    one logical RL step and returns an :class:`~corl.schemas.learning.Environment`
    schema describing the resulting state.

    Two execution modes are supported:

    - **Run mode** (``config["train"]`` is falsy): drives the simulation with
      :meth:`run` until the episode ends or Webots stops.
    - **Training mode** (``config["train"]`` is truthy): drives multi-episode
      training via :meth:`train`, synchronising with a remote agent through the
      :class:`~corl.api.wrapper.Wrapper` API.

    Attributes:
        supervisor: Webots ``Supervisor`` node controlling the simulation.
        timestep: Simulation timestep in milliseconds.
        max_step: Maximum number of logical RL steps per episode.
        train_id: Training session identifier (training mode only).
        worker_id: Worker identifier within the training session (training mode only).
        episode_id: Current episode counter (training mode only).
        request_interval_steps: How often (in simulation timesteps) to poll
            the API for an action acknowledgment (training mode only).
        step_timeout_seconds: Maximum wall-clock seconds to wait for an action
            acknowledgment before triggering the retry handler (training mode only).
        api: :class:`~corl.api.wrapper.Wrapper` instance for API communication
            (training mode only).
    """

    supervisor: Supervisor
    timestep: int
    max_step: int
    _step: int

    # Training attributes
    train_id: str | None = None
    worker_id: int | None = None
    episode_id: int | None = None
    request_interval_steps: int | None = None
    step_timeout_seconds: int | None = None
    api: Wrapper | None = None

    def __init__(
        self,
        supervisor: Supervisor,
        timestep: int,
        max_step: int,
        config: Config,
    ):
        """
        Initialize the environment.

        In run mode (``config["train"]`` is falsy) only the simulation
        attributes are set; all training attributes remain ``None``.

        In training mode (``config["train"]`` is truthy) the training
        attributes are populated and a :class:`~corl.api.wrapper.Wrapper`
        instance is created.

        Args:
            supervisor: Webots ``Supervisor`` node used to control the
                simulation.
            timestep: Simulation timestep in milliseconds passed to
                ``supervisor.step()``.
            max_step: Maximum number of logical RL steps per episode.
            config: Application configuration. Must contain ``"train"`` and,
                when ``"train"`` is truthy, ``"train_id"``, ``"worker_id"``,
                ``"get_request_interval_steps"``, and
                ``"environment_step_timeout"``.
        """
        self.supervisor = supervisor
        self.timestep = timestep
        self.max_step = max_step
        self._step = 0

        if config["train"]:
            self.train_id = config["train_id"]
            self.worker_id = config["worker_id"]
            self.episode_id = 0
            self.request_interval_steps = config["get_request_interval_steps"]
            self.step_timeout_seconds = config["environment_step_timeout"]
            self.api = Wrapper(config)
            logger.debug(
                f"Environment initialized for training with train_id={self.train_id} and worker_id={self.worker_id}"
            )

    @abstractmethod
    def step(self) -> EnvironmentSchema:
        """
        Advance the simulation by one logical RL step.

        Subclasses must override this method to:
            - Update or compute the new environment state.
            - Compute and return the reward for the current transition.
            - Set ``done=True`` on the returned schema when the episode
              should terminate.

        Returns:
            EnvironmentSchema: The updated environment state after the step.

        Raises:
            NotImplementedError: Always, if the subclass does not override
                this method.
        """
        raise NotImplementedError("Method step() not implemented.")

    def reset(self) -> None:
        """
        Reset the simulation and internal counters to the episode start state.

        Calls ``supervisor.simulationReset()`` and sets ``_step`` back to 0.
        Should be called between episodes to ensure a clean starting state.
        """
        self.supervisor.simulationReset()
        self._step = 0
        logger.debug("Environment reset.")

    def quit(self) -> None:
        """
        Terminate the Webots simulation process with exit code 0.

        Calls ``supervisor.simulationQuit(0)``. This is a terminal action —
        the process will exit and cannot be resumed.
        """
        self.supervisor.simulationQuit(0)
        logger.debug("Environment terminated successfully.")

    def run(self) -> None:
        """
        Run the simulation loop until termination.

        Drives the Webots simulation step-by-step. On each iteration,
        :meth:`step` is called to advance the environment. The loop exits
        when either:

        - ``state.done`` is ``True`` (episode ended naturally), or
        - ``supervisor.step()`` returns ``-1`` (Webots simulation stopped).

        After the loop, :meth:`quit` is called to terminate the process.
        """
        total_reward = 0.0

        while self.supervisor.step(self.timestep) != -1:
            state = self.step()
            total_reward += state.reward
            logger.debug(f"Step index: {self._step}, state: {state.model_dump()}")

            if state.done:
                break

            self._step += 1

        logger.info(
            f"Simulation terminated at step {self._step}, reward: {total_reward}"
        )

        self.quit()

    def train(self) -> None:
        """
        Run the full training loop across all episodes.

        Repeatedly calls :meth:`_train_episode` for as long as the remote
        worker is reported as active by the API. Once the worker is no longer
        active, :meth:`quit` is called to terminate the simulation.

        Requires the environment to have been initialized in training mode
        (i.e. ``config["train"]`` was truthy at construction time).
        """
        while self.api.get_worker_status(self.train_id, self.worker_id):
            logger.info(f"Starting episode {self.episode_id}.")
            self._train_episode()

        self.quit()

    def _train_episode(self) -> None:
        """
        Execute a single training episode.

        Runs the Webots simulation loop and synchronises with the remote
        training agent at each logical RL step:

        1. **Action acknowledgment** — polls ``api.get_action()`` every
           ``request_interval_steps`` timesteps until the agent confirms
           the action has been executed.
        2. **Environment step** — calls :meth:`step` to advance the
           simulation and compute the reward.
        3. **State dispatch** — sends the resulting
           :class:`~corl.schemas.learning.Environment` state to the API via
           ``api.send_environment()``.
        4. **Termination check** — if ``state.done`` is ``True``, increments
           the remote episode counter and breaks the loop.

        If the step timeout (``step_timeout_seconds``) is exceeded while
        waiting for an action acknowledgment,
        :meth:`_train_episode_max_retry_trigger` is invoked.

        On normal exit (done or supervisor stop) the episode counter is
        incremented and :meth:`reset` is called to prepare for the next
        episode.
        """
        action_ack: bool = False
        state: EnvironmentSchema | None = None
        total_reward: float = 0.0
        timestep_count: int = 0
        step_timestep_count: int = 0
        start_counter: float = time.perf_counter()
        first_retry_time: float | None = None

        while self.supervisor.step(self.timestep) != -1:
            timestep_count += 1
            step_timestep_count += 1

            if (
                first_retry_time is not None
                and time.time() - first_retry_time > self.step_timeout_seconds
            ):
                self._train_episode_max_retry_trigger()
                return

            # (1) Wait agent action acknowledgment.
            if not action_ack:
                if (step_timestep_count - 1) % self.request_interval_steps == 0:
                    first_retry_time = (
                        time.time() if first_retry_time is None else first_retry_time
                    )

                    action = self.api.get_action(
                        self.train_id, self.worker_id, self.episode_id, self._step
                    )

                    if action is not None and action.executed:
                        action_ack = True
                        first_retry_time = None
                    else:
                        continue
                else:
                    continue

            # (2) Perform environment step and send new state.
            if state is None:
                state = self.step()
                total_reward += state.reward

            # (3) Send environment state.
            if not self.api.send_environment(
                self.train_id,
                self.worker_id,
                self.episode_id,
                self._step,
                state,
            ):
                continue

            # (4) Episode termination.
            if state.done:
                self.api.increment_episode_id(self.train_id, self.worker_id)
                break

            # (5) Prepare for next iteration.
            logger.debug(
                f"Completed step {self._step + 1}/{self.max_step} for episode {self.episode_id} with reward {state.reward}. Duration: {step_timestep_count} timesteps."
            )

            action_ack = False
            state = None
            self._step += 1
            step_timestep_count = 0
            first_retry_time = None

        logger.info(
            f"Simulation loop exited at step {self._step + 1} for episode {self.episode_id} with total reward {total_reward}. Duration: {(time.perf_counter() - start_counter):.6f} seconds and {timestep_count} timesteps."
        )
        self.episode_id += 1
        self.reset()

    def _train_episode_max_retry_trigger(self) -> None:
        """
        Handle a step timeout during a training episode.

        Called by :meth:`_train_episode` when the agent has not acknowledged
        an action within ``step_timeout_seconds``. Behaviour depends on the
        current state:

        - **Worker active, step == 0**: The agent never acknowledged the very
          first action. Logs an info message and returns so the episode can
          be retried from the top of the training loop.
        - **Worker active, step > 0**: A mid-episode timeout is considered a
          fatal error. Logs an error and calls :meth:`quit` to terminate the
          simulation.
        - **Worker inactive**: The remote worker has stopped. Logs an info
          message and returns without quitting so the outer training loop can
          clean up gracefully.
        """
        if self.api.get_worker_status(self.train_id, self.worker_id):
            if self._step == 0:
                logger.info(
                    f"Worker {self.worker_id} failed to acknowledge initial action for episode {self.episode_id}, retrying..."
                )
            else:
                logger.error(
                    f"Exceeded maximum retries  for episode {self.episode_id}, step {self._step}."
                )
                self.quit()
        else:
            logger.info(
                f"Worker {self.worker_id} is no longer active. Terminating episode {self.episode_id} at step {self._step}."
            )
