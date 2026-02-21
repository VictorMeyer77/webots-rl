import logging
import os
import time
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from corl.api.wrapper import Wrapper
from corl.schemas.learning import Action, Observation
from corl.schemas.tracker import StepKey
from corl.trainer.tracker import StepResult, Tracker
from corl.utils.config import Config

logger = logging.getLogger(__name__)


class Trainer(ABC):
    """
    Abstract base class for reinforcement learning trainers.

    Provides the core training loop infrastructure: API communication, worker
    tracking, TensorBoard logging, and step-level data exchange (observations,
    actions, environment results). Subclasses must implement the model-specific
    logic via ``policy``, ``parse_observations``, ``save_model``, and ``run``.

    Attributes:
        model: The neural network or array-based model used to select actions.
        save_model_path: File-system path where the model will be persisted,
            constructed as ``<trainer_model_dir>/<train_id>``.
        max_worker: Maximum number of concurrent workers for this session.
        train_id: Unique identifier of the training session.
        api: API wrapper used to exchange data with the training server.
        tracker: Tracks worker state, step keys, and per-step result buffers.
        tb_writer: TensorBoard summary writer for logging training metrics.
    """

    model: tf.keras.Model | NDArray[np.float32]
    save_model_path: str
    max_worker: int
    train_id: str
    api: Wrapper
    tracker: Tracker
    tb_writer: tf.summary.SummaryWriter

    def __init__(
        self,
        model: tf.keras.Model | NDArray[np.float32],
        config: Config,
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
        self.model = model
        self.train_id = config["train_id"]
        self.save_model_path = os.path.join(config["trainer_model_dir"], self.train_id)
        self.max_worker = config["trainer_max_worker"]
        self._init_api(config)
        self.tracker = Tracker(self.train_id, self.api)
        self._init_tensorboard(config["trainer_tensorboard_path"], self.train_id)
        logger.debug(
            f"Initialized trainer with max_worker={self.max_worker} and model save path {self.save_model_path}"
        )

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

    def _init_tensorboard(self, tb_path: str, train_id: str) -> None:
        """
        Create the TensorBoard file writer for the current training session.

        The log directory is constructed as ``<tb_path>/<train_id>``.

        Args:
            tb_path: Root directory for TensorBoard logs.
            train_id: Training session identifier appended to ``tb_path``.
        """
        tensorboard_dir = os.path.join(tb_path, train_id)
        self.tb_writer = tf.summary.create_file_writer(tensorboard_dir)
        logger.debug(f"TensorBoard logging to {tensorboard_dir}")

    def close(self) -> None:
        """
        Release all resources held by the trainer.

        Flushes and closes the TensorBoard writer, marks all workers as inactive
        via the tracker, and closes the underlying HTTP session.
        """
        logger.debug(f"Closing trainer {self.train_id}")
        self._close_tensorboard()
        self.tracker.close_workers()
        self.api.close()
        logger.debug("Trainer resources have been cleaned up.")

    def _close_tensorboard(self) -> None:
        """
        Flush and close the TensorBoard writer.

        Should be called after training completes to ensure all events are persisted.
        """
        self.tb_writer.flush()
        self.tb_writer.close()
        logger.debug("TensorBoard writer closed")

    @abstractmethod
    def save_model(self) -> None:
        """
        Save the current model to ``save_model_path``.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Method save() not implemented.")

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
        step_keys = [observation[0] for observation in observations]
        observation_batch = np.array(
            [observation[1] for observation in observations], dtype=np.float32
        )
        if observation_batch.shape[0] == 0:
            logger.debug("No observations to process for action selection.")
            return
        action_array = self.policy(observation_batch)
        actions = [
            Action(action=int(action), executed=False) for action in action_array
        ]
        send_actions = list(zip(step_keys, actions))
        if len(step_keys) > 0:
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

        return step_results
