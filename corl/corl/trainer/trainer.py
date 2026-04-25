import logging
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path

import mlflow
import numpy as np
from mlflow import ActiveRun
from numpy.typing import NDArray

from corl.model.model import Model
from corl.schemas.learning import Action, Observation
from corl.schemas.tracker import StepKey
from corl.trainer.tracker import StepResult, Tracker
from corl.trainer.wrapper import Wrapper
from corl.utils.config import Config
from corl.utils.video import generate_training_video

logger = logging.getLogger(__name__)


class Trainer(ABC):
    """
    Abstract base class for reinforcement learning trainers.

    Provides the core training loop infrastructure: API communication, worker
    tracking, and step-level data exchange (observations, actions, environment
    results). Subclasses must implement the model-specific logic via ``policy``,
    ``parse_observations``, ``params``, and ``run``.

    Attributes:
        train_id: Unique identifier of the training session.
        model: The neural network or array-based model used to select actions.
        model_dir: File-system path where model checkpoints are persisted.
        video_dir: Directory where per-episode video frames are written.
        model_checkpoint_frequency: Number of transitions between automatic checkpoints.
        log_metric_frequency: Minimum number of transitions between MLflow metric log calls.
        api: API wrapper used to exchange data with the training server.
        tracker: Tracks worker state, step keys, and per-step result buffers.
    """

    train_id: str

    model_dir: str
    video_dir: str

    model: Model
    model_checkpoint_frequency: int
    log_metric_frequency: int

    api: Wrapper
    mlflow: ActiveRun
    tracker: Tracker

    def __init__(
        self,
        model: Model,
        config: Config,
        model_checkpoint_frequency: int = 10,
    ):
        """
        Initialise the trainer, API connection, and tracker.

        Reads ``train_id`` and ``world_name`` from *config* to set up the MLflow
        experiment, and ``output_dir`` to derive all output sub-directories.

        Args:
            model: The model to train. Can be a Keras model or a raw numpy array
                for table-based methods.
            config: Application configuration. Must contain ``train_id``,
                ``trainer_output_dir``, ``trainer_worker_timeout``,
                ``trainer_mlflow_url``, ``trainer_log_metric_frequency``,
                and ``world_name``.
            model_checkpoint_frequency: Number of transitions between automatic
                model checkpoints. Defaults to 10.
        """
        self.train_id = config.get("train_id")
        self.model = model
        self.model_checkpoint_frequency = model_checkpoint_frequency
        self.log_metric_frequency = config.get("trainer_log_metric_frequency")
        self._init_output_dir(config.get("trainer_output_dir"))
        self._init_api(config)
        self.tracker = Tracker(
            self.train_id, config.get("trainer_worker_timeout"), self.api
        )
        self._init_mlflow(config.get("world_name"), config.get("trainer_mlflow_url"))

    def close(self) -> None:
        """
        Release all resources held by the trainer.

        Executes the following cleanup steps in order:

        1. Generate and upload the full training video (``_generate_video``).
        2. Upload model artefacts and delete the local model directory
           (``_close_model``).
        3. End the MLflow run (``_close_mlflow``).
        4. Mark all workers as inactive via the tracker.
        5. Close the underlying HTTP session.

        Note:
            Each step is called unconditionally. If an earlier step raises,
            subsequent cleanup steps will be skipped. Wrap individual steps
            in ``try/except`` if partial failure resilience is required.
        """
        self._generate_video()
        self._close_model()
        self._close_mlflow()
        self.tracker.close_workers()
        self.api.close()
        logger.debug(f"Trainer for session {self.train_id} closed")

    # Model

    def _close_model(self) -> None:
        """
        Upload final model artefacts to MLflow and delete the local model directory.

        Iterates over files in ``model_dir`` and logs any file whose name does
        **not** contain ``"_ckt_"`` to MLflow under the ``model`` artefact path.
        Checkpoint files (names containing ``"_ckt_"``) are deliberately skipped
        and will be deleted along with the directory.

        Note:
            If the most recent weights were saved as a checkpoint (i.e. the
            filename contains ``"_ckt_"``), they will **not** be uploaded to
            MLflow. Ensure a non-checkpoint save is performed before calling
            ``close()``.
        """
        model_path = Path(self.model_dir)
        for file in model_path.iterdir():
            if "_ckt_" not in file.name:
                mlflow.log_artifact(str(file), artifact_path="model")
                logger.debug(f"Model file {file.name} logged to MLflow")
        shutil.rmtree(self.model_dir)
        logger.debug(f"Model directory {self.model_dir} deleted")

    # Working directory setup

    def _init_output_dir(self, output_dir: str) -> None:
        """
        Derive and create all output sub-directories for this training session.

        Sets the following instance attributes and creates the corresponding
        directories (including any missing parents):

        - ``model_dir``  → ``<output_dir>/models/<train_id>``
        - ``video_dir``  → ``<output_dir>/videos/<train_id>``

        Args:
            output_dir: Root output directory, typically from
                ``config.get("trainer_output_dir")``.
        """
        base = Path(output_dir)
        self.model_dir = str(base / "models" / self.train_id)
        self.video_dir = str(base / "videos" / self.train_id)
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.video_dir).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output directory {output_dir} initialized")

    # Backtrain api

    def _init_api(self, config: Config) -> None:
        """
        Create the API wrapper and register the training session on the server.

        Args:
            config: Application configuration forwarded to ``Wrapper``.

        Raises:
            requests.RequestException: If the HTTP request to create the session fails.
            RuntimeError: If the server returns a non-success response.
        """
        self.api = Wrapper(config)
        self.api.create_training_session(self.train_id)
        logger.debug(f"Initialized API for training session {self.train_id}")

    # MLflow

    def _init_mlflow(self, experiment_name: str, mlflow_url: str) -> None:
        """
        Initialize MLflow tracking for the training session.

        Sets the tracking URI to ``mlflow_url``, creates the experiment if it
        does not already exist, and starts a new MLflow run named after
        ``train_id``.

        Args:
            experiment_name: Name of the MLflow experiment to log runs under.
                Created automatically if it does not already exist.
            mlflow_url: Tracking server URI passed to
                ``mlflow.set_tracking_uri`` (e.g.
                ``"http://localhost:5001"``).
        """

        mlflow.set_tracking_uri(mlflow_url)

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_id=experiment_id)

        run = mlflow.start_run(run_name=self.train_id)
        logger.debug(
            f"MLflow run started with ID {run.info.run_id} for training session {self.train_id}"
        )

    @staticmethod
    def _close_mlflow() -> None:
        """
        End the MLflow run.

        Should be called after training completes to ensure the run is properly closed.
        """
        mlflow.end_run()
        logger.debug("MLflow run ended")

    # Video generation

    def _generate_video(self) -> None:
        """
        Compile episode frames into a video, upload it to MLflow, and clean up.

        Calls :func:`~corl.utils.video.generate_training_video` on ``video_dir``.
        If a video is produced, it is logged to MLflow under the ``videos``
        artefact path. If generation fails for any reason, a warning is emitted
        and the error is suppressed so that the rest of ``close()`` can proceed.
        The ``video_dir`` directory is always removed in a ``finally`` block,
        regardless of success or failure.
        """
        try:
            video_path = generate_training_video(self.video_dir)
            if video_path is not None:
                mlflow.log_artifact(video_path, artifact_path="videos")
                logger.debug(
                    f"Generated full video at {video_path} and uploaded to MLflow"
                )
        except Exception:
            logger.warning("Video generation failed; skipping.", exc_info=True)
        finally:
            shutil.rmtree(self.video_dir, ignore_errors=True)

    # Abstract methods to implement in subclasses

    @abstractmethod
    def params(self) -> dict[str, str | int | float]:
        """
        Return a dictionary of hyperparameters for logging.

        The returned dictionary should contain key-value pairs representing the
        hyperparameters of the training session, such as learning rate, batch
        size, or algorithm-specific parameters. This information is used for
        logging and experiment tracking purposes.

        Returns:
            A dictionary where keys are hyperparameter names and values are their
            corresponding values (string, integer, or float).
        """

    @abstractmethod
    def run(self, epochs: int) -> None:
        """
        Run the full training process for a given number of epochs.

        Args:
            epochs: Number of training epochs to execute.

        Returns:
            None
        """

    @abstractmethod
    def policy(self, observations: NDArray[np.float32]) -> NDArray[np.int32]:
        """
        Select actions for a batch of observations.

        Args:
            observations: Array of shape ``(N, obs_dim)`` containing the current
                observations for ``N`` workers.

        Returns:
            Integer action array of shape ``(N,)``, one action per observation.
        """

    @abstractmethod
    def parse_observations(
        self, observations: list[tuple[StepKey, Observation]]
    ) -> list[tuple[StepKey, NDArray[np.float32]]]:
        """
        Convert raw ``Observation`` objects into numpy arrays suitable for the model.

        Args:
            observations: List of ``(StepKey, Observation)`` pairs received from the
                environment. ``None`` values have already been filtered out before
                this method is called.

        Returns:
            List of ``(StepKey, NDArray[np.float32])`` pairs where each array is the
            numerical representation of the corresponding observation.
        """

    # Training step implementations

    def _training_step_observation(
        self,
    ) -> list[tuple[StepKey, NDArray[np.float32]]]:
        """
        Fetch pending observations from the API and populate the tracker buffer.

        Queries the API for workers whose observation buffer is currently ``None``,
        filters out any ``None`` responses (not yet available), converts the
        ``Observation`` objects to arrays via ``parse_observations``, and stores
        them in the tracker buffer.

        Returns:
            List of ``(StepKey, NDArray[np.float32])`` pairs for every observation
            successfully retrieved and parsed in this call.
        """
        none_step_keys = self.tracker.get_buffer_none_observations()
        observations = self.api.get_observation_batch(self.train_id, none_step_keys)
        observations = [
            observation for observation in observations if observation[1] is not None
        ]
        observation_arrays = self.parse_observations(observations)
        self.tracker.add_buffer_observations(observation_arrays)
        logger.debug(f"Processed {len(observation_arrays)} observations")
        return observation_arrays

    def _training_step_action(
        self,
        observations: list[tuple[StepKey, NDArray[np.float32]]],
    ) -> None:
        """
        Run the policy on the current observations and send the resulting actions to the API.

        Batches all observation arrays, calls ``policy`` once, wraps each result in
        an ``Action`` object, publishes the batch via the API, and stores the actions
        in the tracker buffer.

        Args:
            observations: List of ``(StepKey, NDArray[np.float32])`` pairs as returned
                by ``_training_step_observation``. If empty, this method is a no-op.

        Raises:
            RuntimeError: If ``api.send_action_batch`` returns ``False``, indicating
                that the server did not acknowledge the actions.
        """
        if len(observations) == 0:
            logger.debug("No observations to process for action selection.")
            return
        step_keys = [observation[0] for observation in observations]
        observation_batch = np.array(
            [observation[1] for observation in observations], dtype=np.float32
        )
        action_array = self.policy(observation_batch)
        actions = [
            Action(action=int(action), executed=False) for action in action_array
        ]
        if len(actions) != len(step_keys):
            logger.error(
                f"Policy returned {len(actions)} actions for {len(step_keys)} observations"
            )
            raise RuntimeError(
                f"Policy returned {len(actions)} actions for {len(step_keys)} observations"
            )
        send_actions = list(zip(step_keys, actions))
        if not self.api.send_action_batch(self.train_id, send_actions):
            logger.error(f"Failed to send {len(send_actions)} actions to API")
            raise RuntimeError("Failed to send action batch to API.")
        self.tracker.add_buffer_actions(send_actions)
        logger.debug(f"Sent {len(send_actions)} actions")

    def _training_step_environment(self) -> None:
        """
        Fetch pending environment results from the API and populate the tracker buffer.

        Queries the API for workers whose reward buffer is currently ``None``,
        filters out ``None`` responses, and stores the valid ``Environment``
        objects (reward and done flag) in the tracker buffer.
        """
        step_keys = self.tracker.get_buffer_none_environments()
        environments = self.api.get_environment_batch(self.train_id, step_keys)
        environments = [
            environment for environment in environments if environment[1] is not None
        ]
        self.tracker.add_buffer_environments(environments)
        logger.debug(f"Processed {len(environments)} environment states")

    def training_step(self) -> list[tuple[StepKey, StepResult]]:
        """
        Execute one full training step across all active workers.

        The sequence is:

        1. Refresh the worker list from the API via ``tracker.refresh()``.
        2. Sleep 1 second and return early if no workers are active yet.
        3. Fetch observations (``_training_step_observation``), select actions
           (``_training_step_action``), and fetch environment results
           (``_training_step_environment``).
        4. Collect all workers whose buffers are complete (observation + action
           + reward + done flag).
        5. Advance each completed worker: increment episode on ``done=True``,
           increment step otherwise.
        6. Remove timed-out workers via ``tracker.worker_timeouts()``.

        Returns:
            List of ``(StepKey, StepResult)`` tuples for every worker that
            completed a step in this call. Returns an empty list if no workers
            are currently active.
        """
        self.tracker.refresh()

        if len(self.tracker.worker_step_keys()) == 0:
            time.sleep(1)
            logger.debug("No active workers found for episode step.")
            return []

        observations = self._training_step_observation()
        self._training_step_action(observations)
        self._training_step_environment()

        step_results = self.tracker.get_buffered_step_results()

        for step_key, step_result in step_results:
            if step_result.done:
                self.tracker.increment_episode(step_key.worker_id)
            else:
                self.tracker.increment_step(step_key.worker_id, step_key.episode_id)

        self.tracker.worker_timeouts()

        logger.debug(f"Completed training step with {len(step_results)} step results")

        return step_results
