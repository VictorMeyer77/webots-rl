import logging
import os
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
        timestep_index: Number of simulation timesteps elapsed in the current episode.
        max_timestep: Maximum number of simulation timesteps per episode.
        train_id: Training session identifier (training mode only).
        worker_id: Worker identifier within the training session (training mode only).
        episode_id: Current episode counter (training mode only).
        video_directory: Directory where episode recordings are saved (training mode only).
        is_recording: Whether a video recording is currently in progress (training mode only).
        recording_frequency: Record a video every this many episodes, for worker 0 (training mode only).
        api: :class:`~corl.api.wrapper.Wrapper` instance for API communication
            (training mode only).
    """

    supervisor: Supervisor
    timestep: int
    timestep_index: int
    max_timestep: int

    # Training attributes
    train_id: str | None = None
    worker_id: int | None = None
    episode_id: int | None = None

    # step_timeout_seconds: int | None = None
    video_directory: str | None = None
    is_recording: bool = False
    recording_frequency: int | None = None
    api: Wrapper | None = None

    def __init__(
        self,
        supervisor: Supervisor,
        timestep: int,
        max_timestep: int,
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
            max_timestep: Maximum number of simulation timesteps per episode.
            config: Application configuration. Must contain ``"train"`` and,
                when ``"train"`` is truthy, ``"train_id"``, ``"worker_id"``,
                ``"trainer_output_dir"``, and ``"environment_record_frequency"``.
        """
        self.supervisor = supervisor
        self.timestep = timestep
        self.timestep_index = 0
        self.max_timestep = max_timestep

        if config["train"]:
            self.train_id = config["train_id"]
            self.worker_id = config["worker_id"]
            self.episode_id = 0
            self.video_directory = os.path.join(
                config["trainer_output_dir"], "videos", self.train_id
            )
            self.is_recording = False
            self.recording_frequency = config["environment_record_frequency"]
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
        """
        raise NotImplementedError("Method step() not implemented.")

    def reset(self) -> None:
        """
        Reset the simulation and internal counters to the episode start state.

        Calls ``supervisor.simulationReset()`` and sets ``timestep_index`` back to 0.
        Should be called between episodes to ensure a clean starting state.
        """
        self.supervisor.simulationReset()
        self.timestep_index = 0
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

        self.supervisor.step(self.timestep)

        while self.supervisor.step(self.timestep) != -1:
            state = self.step()
            total_reward += state.reward
            self.timestep_index += 1

            if state.done:
                break

        logger.info(
            f"Simulation terminated in {self.supervisor.getTime() - self.timestep / 1000} seconds with reward: {total_reward}"
        )

        self.quit()

    def train(self, action_repeat: int) -> None:
        """
        Run the full training loop across all episodes.

        Repeatedly calls :meth:`_train_episode` for as long as the remote
        worker is reported as active by the API. Once the worker is no longer
        active, :meth:`quit` is called to terminate the simulation.

        Requires the environment to have been initialized in training mode
        (i.e. ``config["train"]`` was truthy at construction time).
        """
        while self.api.get_worker_status(self.train_id, self.worker_id):
            to_record = (
                self.worker_id == 0
                and self.episode_id > 0
                and self.episode_id % self.recording_frequency == 0
            )

            logger.info(f"Starting episode {self.episode_id}. Recording: {to_record}")

            if to_record and not self.is_recording:
                self._train_start_recording()

            self._train_episode(action_repeat)

            if self.is_recording:
                self._train_stop_recording()

            self.reset()

        self.quit()

    def _train_episode(self, action_repeat: int) -> None:
        """
        Run a single training episode until termination.

        Executes the simulation loop by alternating between
        :meth:`_train_execute_training_step` (advancing the simulation by
        ``action_repeat`` timesteps) and :meth:`_train_evaluate_step`
        (computing the state and sending it to the remote agent). The loop
        exits when ``state.done`` is ``True``.

        After the loop, increments the remote episode counter via the API
        and advances :attr:`episode_id`.

        Args:
            action_repeat: Number of simulation timesteps to advance per
                logical RL step before evaluating the state.
        """
        training_step: int = 0
        total_reward: float = 0.0
        start_counter: float = time.perf_counter()

        self.supervisor.step(self.timestep)

        while True:
            self._train_execute_training_step(training_step, action_repeat)
            state = self._train_evaluate_step(training_step)
            total_reward += state.reward
            training_step += 1
            logger.debug(
                f"Completed training step {training_step} for episode {self.episode_id} with reward {state.reward}."
            )

            if state.done:
                break

        logger.info(
            f"Simulation loop exited at training step {training_step} for episode {self.episode_id} with total reward {total_reward}. Duration: {(time.perf_counter() - start_counter):.6f} seconds and {self.supervisor.getTime()} simulation seconds."
        )
        self.api.increment_episode_id(self.train_id, self.worker_id)
        self.episode_id += 1

    def _train_execute_training_step(
        self, training_step: int, action_repeat: int
    ) -> None:
        """
        Advance the simulation by ``action_repeat`` timesteps for one RL step.

        Calls ``supervisor.step()`` repeatedly until ``action_repeat``
        successful ticks have been completed or Webots signals termination
        (return value ``-1``). Each successful tick increments
        :attr:`timestep_index`.

        Args:
            training_step: Current logical RL step index within the episode,
                used only for debug logging.
            action_repeat: Number of simulation timesteps to execute before
                returning control to the caller.
        """
        action_repeat_count = 0

        while action_repeat_count < action_repeat:
            if self.supervisor.step(self.timestep) != -1:
                action_repeat_count += 1
                self.timestep_index += 1
            else:
                break

        logger.debug(
            f"Completed action repeat for training step {training_step} at timestep {self.supervisor.getTime()} with action_repeat_count {action_repeat_count}."
        )

    def _train_evaluate_step(self, training_step: int) -> EnvironmentSchema:
        """
        Evaluate the current state and send it to the remote agent.

        Calls :meth:`step` to obtain the current environment state, then
        forwards it to the remote agent via
        :meth:`~corl.api.wrapper.Wrapper.send_environment`. If the API call
        fails, logs an error and calls :meth:`quit` to terminate the process.

        Args:
            training_step: Current logical RL step index within the episode,
                forwarded to the API payload.

        Returns:
            EnvironmentSchema: The environment state returned by :meth:`step`.
        """
        state = self.step()
        if not self.api.send_environment(
            self.train_id,
            self.worker_id,
            self.episode_id,
            training_step,
            state,
        ):
            logger.error(
                f"Failed to send environment state for episode {self.episode_id}, training step {training_step}."
            )
            self.quit()

        return state

    def _train_start_recording(self) -> None:
        """
        Start recording the current episode as an MP4 video.

        Calls ``supervisor.movieStartRecording()`` targeting
        ``<video_directory>/episode_<episode_id>.mp4``. Sets
        :attr:`is_recording` to ``True`` on success, or ``False`` if the
        call raises an exception (warning is logged in that case).
        """
        try:
            self.supervisor.movieStartRecording(
                filename=os.path.join(
                    self.video_directory, f"episode_{self.episode_id}.mp4"
                ),
                quality=90,
                caption=False,
                width=800,
                height=600,
                codec=0,
                acceleration=4,
            )
            logger.info("Started recording episode video.")
            self.is_recording = True
        except Exception as e:
            logger.warning(f"Failed to start video recording: {e}")
            self.is_recording = False

    def _train_stop_recording(self) -> None:
        """
        Stop the current episode video recording.

        Calls ``supervisor.movieStopRecording()`` and sets
        :attr:`is_recording` to ``False``. If the call raises an exception,
        the error is logged as a warning and :attr:`is_recording` is still
        set to ``False``.
        """
        try:
            self.supervisor.movieStopRecording()
            logger.info("Stopped recording episode video.")
        except Exception as e:
            logger.warning(f"Failed to stop video recording: {e}")
        self.is_recording = False
