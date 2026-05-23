import logging
import shutil
import time
import zipfile
from abc import ABC, abstractmethod
from datetime import datetime
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
        model_dir: File-system path where final model artefacts are persisted.
        video_dir: Directory where per-episode video frames are written.
        checkpoint_dir: Directory where periodic training checkpoints are saved.
        checkpoint_frequency: Number of transitions between automatic checkpoints.
        log_metric_frequency: Minimum number of transitions between MLflow metric log calls.
        episode_count: Total number of episodes completed across all workers since training started.
        mlflow: The active MLflow run context manager.
        mlflow_run_id: ID of the active MLflow run; persisted across recoveries.
        api: API wrapper used to exchange data with the training server.
        tracker: Tracks worker state, step keys, and per-step result buffers.
    """

    train_id: str

    model_dir: str
    video_dir: str
    checkpoint_dir: str

    model: Model
    checkpoint_frequency: int
    last_checkpoint_transition: int
    log_metric_frequency: int
    episode_count: int

    api: Wrapper
    mlflow: ActiveRun
    mlflow_run_id: str | None
    tracker: Tracker

    def __init__(
        self,
        model: Model,
        config: Config,
        checkpoint_frequency: int = 100_000,
        checkpoint_id: str | None = None,
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
            checkpoint_frequency: Number of transitions between automatic
                model checkpoints. Defaults to 100_000.
            checkpoint_id: If provided, resume training from this checkpoint
                via :meth:`recovery`. When ``None``, a fresh training
                session is started. Defaults to ``None``.
        """
        self.model = model
        self.train_id = config.get("train_id")
        self.checkpoint_frequency = checkpoint_frequency
        self.last_checkpoint_transition = 0
        self.log_metric_frequency = config.get("trainer_log_metric_frequency")
        self.episode_count = 0
        self.mlflow_run_id = None
        self._init_output_dir(config.get("trainer_output_dir"))

        if checkpoint_id:
            self.recovery(checkpoint_id)

        self._init_mlflow(config.get("world_name"), config.get("trainer_mlflow_url"))
        self._init_api(config, checkpoint_id is not None)
        self.tracker = Tracker(
            self.train_id, config.get("trainer_worker_timeout"), self.api
        )

    def close(self) -> None:
        """
        Release all resources held by the trainer.

        Executes the following cleanup steps in order:

        1. Generate and upload the full training video (``_generate_video``).
        2. Upload model artefacts and delete the local model directory
           (``_close_model``).
        3. Zip and upload all checkpoints, then delete the checkpoint directory
           (``_close_checkpoints``).
        4. End the MLflow run (``_close_mlflow``).
        5. Mark all workers as inactive via the tracker.
        6. Close the underlying HTTP session.

        Note:
            Each step is called unconditionally. If an earlier step raises,
            subsequent cleanup steps will be skipped. Wrap individual steps
            in ``try/except`` if partial failure resilience is required.
        """
        self._generate_video()
        self._close_model()
        self._close_checkpoints()
        self._close_mlflow()
        self.tracker.close_workers()
        self.api.close()
        logger.debug(f"Trainer for session {self.train_id} closed")

    # Model

    def _close_model(self) -> None:
        """
        Upload final model artefacts to MLflow and delete the local model directory.

        Iterates over all files in ``model_dir`` and logs each one to MLflow
        under the ``model`` artefact path. Then removes ``model_dir``.
        """
        model_path = Path(self.model_dir)
        for file in model_path.iterdir():
            mlflow.log_artifact(str(file), artifact_path="model")
            logger.debug(f"Model file {file.name} logged to MLflow")
        shutil.rmtree(self.model_dir)
        logger.debug(f"Model directory {self.model_dir} deleted")

    def _close_checkpoints(self) -> None:
        """
        Zip all checkpoints, upload the archive to MLflow, then delete the directory.

        Creates a zip archive named ``checkpoints_<timestamp>.zip`` containing
        the entire ``checkpoint_dir`` tree, logs it to MLflow under the
        ``checkpoints`` artefact path, then removes ``checkpoint_dir`` and
        the zip file.

        If the checkpoint directory is empty or does not exist, this is a no-op.
        """
        checkpoint_path = Path(self.checkpoint_dir)
        if not checkpoint_path.exists() or not any(checkpoint_path.rglob("*")):
            shutil.rmtree(self.checkpoint_dir, ignore_errors=True)
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = checkpoint_path.parent / f"checkpoints_{timestamp}.zip"

        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in checkpoint_path.rglob("*"):
                    if file.is_file():
                        zf.write(file, file.relative_to(checkpoint_path))
            mlflow.log_artifact(str(zip_path), artifact_path="checkpoints")
            logger.debug(f"Checkpoints archived and logged to MLflow: {zip_path.name}")
        except Exception:
            logger.warning("Checkpoint archiving failed; skipping.", exc_info=True)
        finally:
            zip_path.unlink(missing_ok=True)
            shutil.rmtree(self.checkpoint_dir, ignore_errors=True)
            logger.debug(f"Checkpoint directory {self.checkpoint_dir} deleted")

    # Working directory setup

    def _init_output_dir(self, output_dir: str) -> None:
        """
        Derive and create all output sub-directories for this training session.

        Sets the following instance attributes and creates the corresponding
        directories (including any missing parents):

        - ``model_dir``       → ``<output_dir>/models/<train_id>``
        - ``video_dir``       → ``<output_dir>/videos/<train_id>``
        - ``checkpoint_dir``  → ``<output_dir>/checkpoints/<train_id>``

        Args:
            output_dir: Root output directory, typically from
                ``config.get("trainer_output_dir")``.
        """
        base = Path(output_dir)
        self.model_dir = str(base / "models" / self.train_id)
        self.video_dir = str(base / "videos" / self.train_id)
        self.checkpoint_dir = str(base / "checkpoints" / self.train_id)
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.video_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output directory {output_dir} initialized")

    # Backtrain api

    def _init_api(self, config: Config, recover: bool = False) -> None:
        """
        Create the API wrapper and register the training session on the server.

        Args:
            config: Application configuration forwarded to ``Wrapper``.
            recover: If ``True``, the existing session is deleted before a new
                one is created, so the server starts with a clean state for the
                resumed run. Defaults to ``False``.

        Raises:
            requests.RequestException: If the HTTP request to create the session fails.
            RuntimeError: If the server returns a non-success response.
        """
        self.api = Wrapper(config)
        if recover:
            self.api.delete_training_session(self.train_id)
        self.api.create_training_session(self.train_id)
        logger.debug(f"Initialized API for training session {self.train_id}")

    # MLflow

    def _init_mlflow(self, experiment_name: str, mlflow_url: str) -> None:
        """
        Initialize MLflow tracking for the training session.

        Sets the tracking URI to ``mlflow_url``, creates the experiment if it
        does not already exist, then either resumes an existing MLflow run
        (when ``mlflow_run_id`` is already set, e.g. after :meth:`recovery`)
        or starts a new run named after ``train_id``.

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

        if self.mlflow_run_id is not None:
            run = mlflow.start_run(run_id=self.mlflow_run_id)
        else:
            run = mlflow.start_run(run_name=self.train_id)
            self.mlflow_run_id = run.info.run_id

        logger.debug(
            f"MLflow run started with ID {self.mlflow_run_id} for training session {self.train_id}"
        )

    @staticmethod
    def _close_mlflow() -> None:
        """
        End the MLflow run.

        Should be called after training completes to ensure the run is properly closed.
        """
        mlflow.end_run()
        logger.debug("MLflow run ended")

    def _mlflow_log_train_params(self) -> None:
        """
        Log training hyperparameters to the active MLflow run.

        Calls :meth:`params` to retrieve the hyperparameter dictionary, then
        filters out string values before passing the remainder to
        ``mlflow.log_params``. Intended to be called by subclasses at the
        start of training (e.g. on the first transition of a fresh run).
        """
        params = {k: v for k, v in self.params().items() if not isinstance(v, str)}
        mlflow.log_params(params)

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

    def params(self) -> dict[str, str | int | float | bool]:
        """
        Return a dictionary of hyperparameters for logging.

        The default implementation collects all instance attributes whose
        values are ``int``, ``float``, ``str``, or ``bool`` via ``vars(self)``.
        Subclasses may override this to expose only the relevant parameters
        or to add algorithm-specific entries.

        Note:
            :meth:`_mlflow_log_train_params` will further filter out string
            values before uploading to MLflow.

        Returns:
            A dictionary where keys are hyperparameter names and values are their
            corresponding values (string, integer, float, or bool).
        """
        return {
            k: v
            for k, v in (vars(self)).items()
            if isinstance(v, (int, float, str, bool))
        }

    @abstractmethod
    def run(self, max_transitions: int) -> None:
        """
        Run the full training process for a given number of transitions.

        Args:
            max_transitions: Total number of transitions to process before stopping.
        """

    @abstractmethod
    def policy(self, observations: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Select actions for a batch of observations.

        Args:
            observations: Array of shape ``(N, obs_dim)`` containing the current
                observations for ``N`` workers.

        Returns:
            Action array of shape ``(N,)`` for discrete actions or ``(N, action_dim)``
            for continuous actions. Each row is converted to ``list[float]`` before
            being wrapped in an ``Action`` object.
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

    @abstractmethod
    def checkpoint(self) -> None:
        """
        Persist all training state to prevent data loss on failure.

        Implementations must save model weights, optimizer states, and all
        serialisable instance attributes needed to resume training, including
        ``last_checkpoint_transition``.
        """

    @abstractmethod
    def recovery(self, checkpoint_id: str) -> None:
        """
        Restore training state from a checkpoint.

        Implementations must reload model weights, optimizer states, and all
        serialisable instance attributes (including ``last_checkpoint_transition``) so
        that training can resume seamlessly from the saved point.

        Args:
            checkpoint_id: Identifier of the checkpoint to restore from,
                as produced by :meth:`checkpoint`.
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
            RuntimeError: If the number of actions returned by ``policy`` does
                not match the number of input observations.
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
            Action(action=np.atleast_1d(a).astype(float).tolist()) for a in action_array
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
            logger.debug("No active workers found.")
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
