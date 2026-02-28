import logging
import os
import shutil
import time
from abc import ABC, abstractmethod

import mlflow
import numpy as np
import tensorflow as tf
from mlflow import ActiveRun
from numpy.typing import NDArray

from corl.api.wrapper import Wrapper
from corl.schemas.learning import Action, Observation
from corl.schemas.tracker import StepKey
from corl.trainer.tracker import StepResult, Tracker
from corl.utils.config import Config
from corl.utils.video import generate_training_video

logger = logging.getLogger(__name__)


class Trainer(ABC):
    """
    Abstract base class for reinforcement learning trainers.

    Provides the core training loop infrastructure: API communication, worker
    tracking, TensorBoard logging, and step-level data exchange (observations,
    actions, environment results). Subclasses must implement the model-specific
    logic via ``policy``, ``parse_observations``, ``save_model``, and ``run``.

    Attributes:
        train_id: Unique identifier of the training session.
        experiment_name: Name of the MLflow experiment this session belongs to.
        model: The neural network or array-based model used to select actions.
        model_dir: File-system path where model checkpoints are persisted.
        tensorboard_dir: Directory for TensorBoard event files.
        mlflow_dir: Root directory for the local MLflow SQLite database and artifacts.
        video_dir: Directory where per-episode video frames are written.
        model_checkpoint_frequency: Number of epochs between automatic checkpoints.
        model_checkpoint_index: Counter tracking how many checkpoints have been saved.
        api: API wrapper used to exchange data with the training server.
        mlflow: Active MLflow run for the current training session.
        tracker: Tracks worker state, step keys, and per-step result buffers.
        tb_writer: TensorBoard summary writer for logging training metrics.
    """

    train_id: str
    experiment_name: str

    model_dir: str
    tensorboard_dir: str
    mlflow_dir: str
    video_dir: str

    model: tf.keras.Model | NDArray[np.float32] | None
    model_checkpoint_frequency: int
    model_checkpoint_index: int

    api: Wrapper
    mlflow: ActiveRun
    tracker: Tracker
    tb_writer: tf.summary.SummaryWriter

    def __init__(
        self,
        model: tf.keras.Model | NDArray[np.float32] | None,
        train_id: str,
        experiment_name: str,
        config: Config,
        model_checkpoint_frequency: int = 10,
    ):
        """
        Initialise the trainer, API connection, tracker, and TensorBoard writer.

        Args:
            model: The model to train. Can be a Keras model or a raw numpy array
                for table-based methods.
            config: Application configuration. Must contain the keys
                ``train_id``, ``trainer_model_dir``, ``trainer_max_worker``,
                ``trainer_tensorboard_path``, ``api_host``, and ``api_port``.
        """
        self.train_id = train_id
        self._init_model(model, model_checkpoint_frequency)
        self._init_output_dir(config)
        self._init_api(config)
        self.tracker = Tracker(train_id, config, self.api)
        self._init_mlflow(experiment_name)
        self._init_tensorboard()

    def close(self) -> None:
        """
        Release all resources held by the trainer.

        Flushes and closes the TensorBoard writer, marks all workers as inactive
        via the tracker, and closes the underlying HTTP session.
        """
        self._generate_video()
        self._close_tensorboard()
        self._close_mlflow()
        self._delete_model_checkpoints()
        self.tracker.close_workers()
        self.api.close()
        logger.debug(f"Trainer for session {self.train_id} closed")

    # Model

    def _init_model(
        self,
        model: tf.keras.Model | NDArray[np.float32] | None,
        model_checkpoint_frequency: int,
    ) -> None:
        self.model = model
        self.model_checkpoint_frequency = model_checkpoint_frequency
        self.model_checkpoint_index = 0
        logger.debug(
            f"Model initialized with checkpoint frequency {model_checkpoint_frequency}"
        )

    def _delete_model_checkpoints(self) -> None:
        shutil.rmtree(self.model_dir)
        logger.debug(f"Deleted existing model checkpoints in {self.model_dir}")

    # Working directory setup

    def _init_output_dir(self, config: Config) -> None:
        output_dir = config["trainer_output_dir"]
        self.model_dir = os.path.join(output_dir, "models", self.train_id)
        self.tensorboard_dir = os.path.join(output_dir, "tensorboard", self.train_id)
        self.mlflow_dir = os.path.join(output_dir, "mlflow")
        self.video_dir = os.path.join(output_dir, "videos", self.train_id)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        os.makedirs(self.mlflow_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
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

    # Tensorboard

    def _init_tensorboard(self) -> None:
        """
        Create the TensorBoard file writer for the current training session.

        The log directory is constructed as ``<tensorboard_dir>/<train_id>``.
        """
        self.tb_writer = tf.summary.create_file_writer(self.tensorboard_dir)
        logger.debug(f"TensorBoard logging to {self.tensorboard_dir}")

    def _close_tensorboard(self) -> None:
        """
        Flush and close the TensorBoard writer.

        Should be called after training completes to ensure all events are persisted.
        """
        self.tb_writer.flush()
        self.tb_writer.close()
        mlflow.log_artifacts(self.tensorboard_dir, artifact_path="tensorboard")
        shutil.rmtree(self.tensorboard_dir)
        logger.debug("TensorBoard writer closed and logs uploaded to MLflow")

    # MLflow

    def _init_mlflow(self, experiment_name: str) -> None:
        """
        Initialize MLflow tracking for the training session.

        Sets the tracking URI to a local SQLite database, creates the experiment
        if it does not already exist, and starts a new run named after
        ``train_id``.

        Args:
            experiment_name: Name of the MLflow experiment to log runs under.
        """

        mlflow.set_tracking_uri(
            f"sqlite:///{os.path.join(self.mlflow_dir, 'mlflow.db')}"
        )

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=f"file:{self.mlflow_dir}",
            )
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
        try:
            video_path = generate_training_video(self.video_dir)
            mlflow.log_artifact(video_path, artifact_path="videos")
            logger.debug(f"Generated full video at {video_path} and uploaded to MLflow")
        except Exception:
            logger.warning("Video generation failed; skipping.", exc_info=True)
        finally:
            shutil.rmtree(self.video_dir, ignore_errors=True)

    # Abstract methods to implement in subclasses

    @abstractmethod
    def save_model(self, checkpoint: bool = False) -> None:
        """
        Persist the current model to disk.

        Args:
            checkpoint: If ``True``, save as an intermediate checkpoint rather
                than overwriting the final model file.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Method save() not implemented.")

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

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """

        raise NotImplementedError("Method params() not implemented.")

    @abstractmethod
    def run(self, epochs: int) -> None:
        """
        Run the full training process for a given number of epochs.

        Args:
            epochs: Number of training epochs to execute.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Method run() not implemented.")

    @abstractmethod
    def policy(self, observations: NDArray[np.float32]) -> NDArray[np.int32]:
        """
        Select actions for a batch of observations.

        Args:
            observations: Array of shape ``(N, obs_dim)`` containing the current
                observations for ``N`` workers.

        Returns:
            Integer action array of shape ``(N,)``, one action per observation.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Method policy() not implemented.")

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

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Method parse_observations() not implemented.")

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
        1. Refresh the worker list from the API (rate-limited by ``REFRESH_RATE``).
        2. Sleep and return early if no workers are active yet.
        3. Fetch observations, select actions, and fetch environment results.
        4. Collect all workers whose buffers are complete (observation + action +
           reward + done).
        5. Advance each completed worker: increment episode on ``done=True``,
           increment step otherwise.
        6. Remove timed-out workers.

        Returns:
            List of ``(StepKey, StepResult)`` tuples for every worker that completed
            a step in this call. Returns an empty list if no workers are active.
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
