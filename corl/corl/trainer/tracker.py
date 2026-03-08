import logging
import time

import numpy as np
from numpy.typing import NDArray

from corl.schemas.learning import Action, Environment
from corl.schemas.tracker import StepKey, StepResult
from corl.trainer.wrapper import Wrapper

logger = logging.getLogger(__name__)

REFRESH_RATE = 5  # seconds between API refreshes to update worker list


class Tracker:
    """
    Tracks the state of all active workers across episodes and steps for a training session.

    The Tracker is the central book-keeping component of the trainer. It maintains a
    live map of workers to their current ``StepKey`` (worker_id / episode_id / step),
    buffers incoming observations, actions, and environment results until a step is
    complete, and handles worker lifecycle events (registration, timeout, and removal).

    Lifecycle:
        1. ``refresh()`` — periodically syncs the worker list from the API.
        2. ``add_buffer_*()`` — accumulates observations, actions, and environment
           results as they arrive asynchronously from workers and the environment.
        3. ``get_buffered_step_results()`` — yields complete steps ready for training.
        4. ``increment_step()`` / ``increment_episode()`` — advances a worker's position
           and clears its buffer for the next step.
        5. ``worker_timeouts()`` / ``close_workers()`` — removes stale or finished workers.

    Attributes:
        train_id: Identifier of the training session this tracker belongs to.
        worker_timeout: Seconds of inactivity after which a worker is considered timed out.
        api: API wrapper used to query and update worker state on the server.
        _workers: Maps worker_id → current StepKey for each active worker.
        _buffer_results: Maps worker_id → in-progress StepResult for the current step.
        _worker_last_update: Maps worker_id → timestamp of the last buffer update,
            used to detect timed-out workers.
        _last_worker_refresh: Timestamp of the last ``api.get_workers`` call,
            used to throttle refresh requests.
    """

    train_id: str
    worker_timeout: int
    api: Wrapper
    _workers: dict[int, StepKey]
    _last_worker_refresh: float
    _worker_last_update: dict[int, float]
    _buffer_results: dict[int, StepResult]

    def __init__(self, train_id: str, worker_timeout: int, api: Wrapper):
        """
        Initialise the tracker for a training session.

        Args:
            train_id (str): Unique identifier of the training session to track.
            worker_timeout (int): Seconds of inactivity after which a worker is
                considered timed out and removed.
            api (Wrapper): API wrapper used to query worker state and push
                status updates to the server.
        """
        self.train_id = train_id
        self.worker_timeout = worker_timeout
        self.api = api
        self._workers = {}
        self._worker_last_update = {}
        self._last_worker_refresh = 0
        self._buffer_results = {}

    def worker_step_keys(self) -> list[StepKey]:
        """
        Return the current StepKey for every tracked worker.

        Returns:
            List of StepKey objects reflecting each worker's current
            worker_id, episode_id, and step.
        """
        return list(self._workers.values())

    def refresh(self) -> None:
        """
        Synchronise the tracked worker list with the API.

        Polls ``api.get_workers`` at most once every ``REFRESH_RATE`` seconds.
        Active workers not yet tracked are added and given a fresh buffer.
        Workers that have become inactive are removed from the tracker.
        """
        if time.time() - self._last_worker_refresh > REFRESH_RATE:
            workers = self.api.get_workers(self.train_id)
            logger.debug(
                f"Refreshing workers for training {self.train_id}: {len(workers)} workers found"
            )

            for worker in workers:
                worker_id = int(worker["id"])
                status = bool(worker["status"])

                if status and worker_id not in self._workers:
                    self._workers[worker_id] = StepKey(
                        worker_id=worker_id, episode_id=0, step=0
                    )
                    self._reset_buffer(worker_id)
                    logger.info(f"Added worker {worker_id} to tracker")

                elif not status and worker_id in self._workers:
                    self._delete_worker(worker_id)
                    continue

            self._last_worker_refresh = time.time()

    def increment_step(self, worker_id: int, episode_id: int) -> None:
        """
        Advance the step counter for a worker and reset its buffer.

        Args:
            worker_id: ID of the worker to advance.
            episode_id: Expected current episode ID for the worker; used to
                guard against stale updates from a previous episode.

        Raises:
            ValueError: If ``worker_id`` is not tracked.
            ValueError: If ``episode_id`` does not match the worker's current episode.
        """
        self._validate_worker_id(worker_id)
        self._validate_episode_id(worker_id, episode_id)
        current = self._workers[worker_id]
        self._workers[worker_id] = StepKey(
            worker_id=current.worker_id,
            episode_id=current.episode_id,
            step=current.step + 1,
        )
        self._reset_buffer(worker_id)
        logger.debug(
            f"Incremented step for worker {worker_id} to {self._workers[worker_id].step}"
        )

    def increment_episode(self, worker_id: int) -> None:
        """
        Advance the episode counter for a worker, reset its step to 0, and clear its buffer.

        Args:
            worker_id: ID of the worker to advance.

        Raises:
            ValueError: If ``worker_id`` is not tracked.
        """
        self._validate_worker_id(worker_id)
        current = self._workers[worker_id]
        self._workers[worker_id] = StepKey(
            worker_id=current.worker_id, episode_id=current.episode_id + 1, step=0
        )
        self._reset_buffer(worker_id)
        logger.info(
            f"Incremented episode for worker {worker_id} to episode {self._workers[worker_id].episode_id}"
        )

    def get_buffer_none_observations(self) -> list[StepKey]:
        """
        Get list of StepKeys for workers that have no buffered observation.

        Returns:
            List of StepKey objects for workers where the observation buffer is None,
            indicating that an observation has not yet been received for the current step.
        """
        return [
            step_key
            for worker_id, step_key in self._workers.items()
            if self._buffer_results[worker_id].observation is None
        ]

    def get_buffer_none_environments(self) -> list[StepKey]:
        """
        Get list of StepKeys for workers that have no buffered environment result.

        Returns:
            List of StepKey objects for workers where the reward buffer is None,
            indicating that an environment result has not yet been received for the current step.
        """
        return [
            step_key
            for worker_id, step_key in self._workers.items()
            if self._buffer_results[worker_id].reward is None
        ]

    def add_buffer_actions(self, actions: list[tuple[StepKey, Action]]) -> None:
        """
        Add a batch of actions to the buffer for their respective workers.

        Args:
            actions: List of (StepKey, Action) tuples to add to the buffer
        """
        for step_key, action in actions:
            worker_id = step_key.worker_id
            self._validate_worker_id(worker_id)
            if action is not None:
                self._buffer_results[worker_id].action = action.action
                self._worker_last_update[worker_id] = time.time()
            else:
                self._add_none_buffer_error(worker_id, step_key, "action")

    def add_buffer_observations(
        self,
        observations: list[tuple[StepKey, NDArray[np.float32]]],
    ) -> None:
        """
        Add a batch of observations to the buffer for their respective workers.

        Args:
            observations: List of (StepKey, NDArray[np.float32]) tuples to add to the buffer
        """
        for step_key, observation in observations:
            worker_id = step_key.worker_id
            self._validate_worker_id(worker_id)
            if observation is not None:
                self._buffer_results[worker_id].observation = observation
                self._worker_last_update[worker_id] = time.time()
            else:
                self._add_none_buffer_error(worker_id, step_key, "observation")

    def add_buffer_environments(
        self, environments: list[tuple[StepKey, Environment]]
    ) -> None:
        """
        Add a batch of environment results to the buffer for their respective workers.

        Args:
            environments: List of (StepKey, Environment) tuples to add to the buffer
        """
        for step_key, environment in environments:
            worker_id = step_key.worker_id
            self._validate_worker_id(worker_id)
            if environment is not None:
                self._buffer_results[worker_id].reward = environment.reward
                self._buffer_results[worker_id].done = environment.done
                self._worker_last_update[worker_id] = time.time()
            else:
                self._add_none_buffer_error(worker_id, step_key, "environment")

    def get_buffered_step_results(self) -> list[tuple[StepKey, StepResult]]:
        """
        Retrieve all buffered step results that are fully complete for the current step.

        Returns:
            List of (StepKey, StepResult) tuples for workers that have all four fields
            populated in the buffer (observation, action, reward, and done). Workers
            where any field is still None are excluded as their step is still in progress
            and not ready for training.
        """
        results = [
            (step_key, self._buffer_results[worker_id])
            for worker_id, step_key in self._workers.items()
            if self._buffer_results[worker_id].is_complete()
        ]

        return results

    def worker_timeouts(self) -> None:
        """
        Remove workers that have exceeded ``worker_timeout`` seconds without an update.

        Iterates over all tracked workers, collects those whose last recorded update
        is older than the timeout threshold, then calls ``close_workers`` once with
        the full list. Timed-out workers are logged as warnings before removal.
        """
        current_time = time.time()
        workers_to_remove = []
        for worker_id, last_update in self._worker_last_update.items():
            if current_time - last_update > self.worker_timeout:
                workers_to_remove.append(worker_id)
                logger.warning(
                    f"Worker {worker_id} has timed out (no updates for {current_time - last_update:.1f}s)"
                )
        self.close_workers(workers_to_remove)

    def close_workers(self, worker_ids: list[int] | None = None) -> None:
        """
        Mark workers as inactive via the API and remove them from the tracker.

        Args:
            worker_ids: List of worker IDs to close. If None, all currently
                tracked workers are closed.

        Note:
            API errors when marking a worker inactive are logged but do not
            prevent the worker from being removed from the local tracker state.
        """
        worker_ids = list(self._workers.keys()) if worker_ids is None else worker_ids

        for worker_id in worker_ids:
            try:
                self.api.update_worker_status(self.train_id, worker_id, False)
                logger.debug(
                    f"Marked worker {worker_id} as inactive during trainer cleanup."
                )
            except Exception as e:
                logger.error(
                    f"Failed to update worker {worker_id} status during cleanup: {e}"
                )
            finally:
                self._delete_worker(worker_id)

    def _reset_buffer(self, worker_id: int) -> None:
        """
        Reset the action, observation, and environment buffers for a worker.

        Called when a worker starts a new episode or when a step is completed to clear
        out old data.

        Args:
            worker_id: ID of the worker whose buffers should be reset
        """
        self._buffer_results[worker_id] = StepResult()
        self._worker_last_update[worker_id] = time.time()

    def _delete_worker(self, worker_id: int) -> None:
        """
        Remove a worker and all its associated state from the tracker.

        Deletes the worker's entry from ``_workers``, ``_worker_last_update``,
        and ``_buffer_results``.

        Args:
            worker_id: ID of the worker to delete.
        """
        del self._workers[worker_id]
        del self._worker_last_update[worker_id]
        del self._buffer_results[worker_id]
        logger.info(f"Removed worker {worker_id} from tracker")

    def _validate_worker_id(self, worker_id: int) -> None:
        """
        Validate that a worker_id exists in the tracker.

        Args:
            worker_id: Worker ID to validate

        Raises:
            ValueError: If worker_id not found in tracker
        """
        if worker_id not in self._workers:
            logger.error(f"Worker {worker_id} not found in tracker")
            raise ValueError(f"Worker {worker_id} not found in tracker.")

    def _validate_episode_id(self, worker_id: int, episode_id: int) -> None:
        """
        Validate that an episode_id matches the worker's current episode.

        Args:
            worker_id: Worker ID to check
            episode_id: Expected episode ID

        Raises:
            ValueError: If episode_id doesn't match worker's current episode
        """
        if episode_id != self._workers[worker_id].episode_id:
            logger.error(
                f"Episode ID mismatch for worker {worker_id}: expected {self._workers[worker_id].episode_id}, got {episode_id}"
            )
            raise ValueError(
                f"Episode ID mismatch for worker {worker_id}: expected {self._workers[worker_id].episode_id}, got {episode_id}."
            )

    @staticmethod
    def _add_none_buffer_error(
        worker_id: int, step_key: StepKey, data_type: str
    ) -> None:
        """
        Log an error and raise ``ValueError`` when a ``None`` value is written to the buffer.

        Called by ``add_buffer_actions``, ``add_buffer_observations``, and
        ``add_buffer_environments`` when the incoming value is ``None`` instead
        of a valid data object.

        Args:
            worker_id (int): ID of the worker that produced the ``None`` value.
            step_key (StepKey): Current step key for the worker, used in the
                error message.
            data_type (str): Human-readable label for the data type that is
                missing (e.g. ``"action"``, ``"observation"``, ``"environment"``).

        Raises:
            ValueError: Always raised with a message identifying the worker,
                episode, step, and data type.
        """
        logger.error(
            f"Received None {data_type} for worker {worker_id}, episode {step_key.episode_id}, step {step_key.step}"
        )
        raise ValueError(
            f"Received None {data_type} for worker {worker_id}, "
            f"episode {step_key.episode_id}, step {step_key.step}"
        )
