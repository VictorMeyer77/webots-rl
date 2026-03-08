import logging
import os
import time

from corl.api.wrapper import Wrapper
from corl.environment import Environment as BaseEnvironment
from corl.schemas.learning import Environment as EnvironmentSchema
from corl.utils.config import Config

logger = logging.getLogger(__name__)


class TrainerEnvironment:
    """
    Bridge between a Webots :class:`~corl.environment.Environment` and a remote RL trainer.

    Drives the environment through successive training episodes by exchanging
    state information with the remote trainer over HTTP via
    :class:`~corl.api.wrapper.Wrapper`. Each episode follows the pattern:

    1. **Warm-up** – one unchecked ``supervisor.step()`` to initialise sensors.
    2. **Loop** – alternate :meth:`execute_training_step` (advance simulation)
       and :meth:`evaluate_step` (compute & send state) until ``done``.
    3. **Bookkeeping** – increment the remote episode counter and reset the
       simulation for the next episode.

    Optionally records episodes as MP4 video files. Recording is restricted
    to ``worker_id == 0`` and triggered every ``recording_frequency`` episodes.

    Attributes:
        environment: The :class:`~corl.environment.Environment` instance
            being driven.
        train_id: Unique identifier for the current training run.
        worker_id: Index of this worker within the training run.
        episode_id: Local episode counter, incremented after each episode.
        video_directory: Directory where episode MP4 recordings are saved.
        is_recording: Whether a video recording is currently active.
        recording_frequency: Number of episodes between recordings.
        api: HTTP client used to communicate with the remote trainer.
    """

    train_id: str
    worker_id: int
    episode_id: int
    video_directory: str
    is_recording: bool
    recording_frequency: int
    api: Wrapper

    def __init__(
        self,
        environment: BaseEnvironment,
        config: Config,
    ):
        """
        Initialise the training wrapper.

        Args:
            environment: The :class:`~corl.environment.Environment` instance
                to drive during training.
            config: Application configuration. Must contain ``"train_id"``,
                ``"worker_id"``, ``"trainer_output_dir"``, and
                ``"environment_record_frequency"``.
        """
        self.environment = environment
        self.train_id = config.get("train_id")
        self.worker_id = config.get("worker_id")
        self.episode_id = 0
        self.video_directory = os.path.join(
            config.get("trainer_output_dir"), "videos", self.train_id
        )
        os.makedirs(self.video_directory, exist_ok=True)
        self.is_recording = False
        self.recording_frequency = config.get("environment_record_frequency")
        self.api = Wrapper(config)

    def run(self, action_repeat: int) -> None:
        """
        Run the full training loop across all episodes.

        Repeatedly calls :meth:`episode` for as long as the remote worker is
        reported as active by the API. Per episode:

        - Recording is started if ``worker_id == 0`` and
          ``episode_id % recording_frequency == 0`` and not already recording.
        - :meth:`stop_recording` is called *before* ``environment.reset()``;
          Webots may not have fully flushed the video file by the time the
          next episode begins.

        Once the worker is no longer active, ``environment.quit()`` is called
        to terminate the simulation.

        Args:
            action_repeat: Passed directly to :meth:`episode`.
        """
        while self.api.get_worker_status(self.train_id, self.worker_id):
            to_record = (
                self.worker_id == 0 and self.episode_id % self.recording_frequency == 0
            )

            logger.info(f"Starting episode {self.episode_id}. Recording: {to_record}")

            if to_record and not self.is_recording:
                self.start_recording()

            self.episode(action_repeat)

            if self.is_recording:
                self.stop_recording()

            self.environment.reset()

        self.environment.quit()

    def episode(self, action_repeat: int) -> None:
        """
        Run a single training episode until termination.

        Performs one warm-up ``supervisor.step()`` call before entering the
        main loop to allow sensors to initialise (its return value is not
        checked). Then alternates between :meth:`execute_training_step`
        (advancing the simulation by ``action_repeat`` timesteps) and
        :meth:`evaluate_step` (computing the state and sending it to the
        remote agent). The loop exits when any of the following conditions
        are met:

        - ``state.done`` is ``True`` (episode ended naturally), or
        - :meth:`execute_training_step` returns ``False`` (simulator stopped).

        After the loop, increments the remote episode counter via the API
        and advances :attr:`episode_id`.

        Args:
            action_repeat: Number of simulation timesteps to advance per
                logical RL step before evaluating the state.
        """
        training_step: int = 0
        total_reward: float = 0.0
        start_counter: float = time.perf_counter()

        self.environment.supervisor.step(self.environment.timestep)

        while self.execute_training_step(training_step, action_repeat):
            state = self.evaluate_step(training_step)
            total_reward += state.reward
            training_step += 1
            logger.debug(
                f"Completed training step {training_step} for episode {self.episode_id} with reward {state.reward}."
            )

            if state.done:
                break

        logger.info(
            f"Simulation loop exited at training step {training_step} for episode {self.episode_id} with total reward {total_reward}. Duration: {(time.perf_counter() - start_counter):.6f} seconds and {self.environment.supervisor.getTime()} sim-seconds."
        )
        self.api.increment_episode_id(self.train_id, self.worker_id)
        self.episode_id += 1

    def execute_training_step(self, training_step: int, action_repeat: int) -> bool:
        """
        Advance the simulation by ``action_repeat`` timesteps for one RL step.

        Calls ``supervisor.step()`` repeatedly until ``action_repeat``
        successful ticks have been completed or Webots signals termination
        (return value ``-1``). Each successful tick increments
        ``environment.timestep_index``.

        Args:
            training_step: Current logical RL step index within the episode,
                used only for debug logging.
            action_repeat: Number of simulation timesteps to execute before
                returning control to the caller.

        Returns:
            bool: ``True`` if all ``action_repeat`` ticks completed
            successfully; ``False`` if Webots signalled termination
            (``supervisor.step()`` returned ``-1``).
        """
        action_repeat_count = 0

        while action_repeat_count < action_repeat:
            if self.environment.supervisor.step(self.environment.timestep) != -1:
                action_repeat_count += 1
                self.environment.timestep_index += 1
            else:
                return False

        logger.debug(
            f"Completed action repeat for training step {training_step} at {self.environment.supervisor.getTime()} with action_repeat_count {action_repeat_count}."
        )
        return True

    def evaluate_step(self, training_step: int) -> EnvironmentSchema:
        """
        Evaluate the current state and send it to the remote agent.

        Calls ``environment.step()`` to obtain the current environment state,
        then forwards it to the remote agent via
        :meth:`~corl.api.wrapper.Wrapper.send_environment`. If the API call
        fails, logs an error and calls ``environment.quit()`` to terminate
        the process.

        Args:
            training_step: Current logical RL step index within the episode,
                forwarded to the API payload.

        Returns:
            EnvironmentSchema: The environment state returned by
            ``environment.step()``.
        """
        state = self.environment.step()
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
            self.environment.quit()

        return state

    def start_recording(self) -> None:
        """
        Start recording the current episode as an MP4 video.

        Calls ``supervisor.movieStartRecording()`` targeting
        ``<video_directory>/episode_<episode_id>.mp4``. Sets
        :attr:`is_recording` to ``True`` on success, or ``False`` if the
        call raises an exception (warning is logged in that case).
        """
        try:
            self.environment.supervisor.movieStartRecording(
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

    def stop_recording(self) -> None:
        """
        Stop the current episode video recording.

        Calls ``supervisor.movieStopRecording()`` and sets
        :attr:`is_recording` to ``False``. If the call raises an exception,
        the error is logged as a warning and :attr:`is_recording` is still
        set to ``False``.
        """
        try:
            self.environment.supervisor.movieStopRecording()
            logger.info("Stopped recording episode video.")
        except Exception as e:
            logger.warning(f"Failed to stop video recording: {e}")
        self.is_recording = False
