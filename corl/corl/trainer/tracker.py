"""
Training Tracker for Reinforcement Learning Workers

This module provides infrastructure for tracking and synchronizing multiple RL workers
during distributed training. It manages worker lifecycle, episode progression, and
buffers step results for creating state transitions.

Overview:
    The Tracker class coordinates multiple workers executing episodes in parallel,
    maintaining their current state (worker_id, episode_id, step) and buffering
    observations, actions, and rewards to construct (s, a, r, s', done) transitions.

Architecture:
    - Workers are dynamically discovered via API refresh
    - Each worker maintains independent episode/step counters
    - Step results are buffered with maxlen=2 to create transitions
    - Transitions are yielded when buffer contains adjacent steps

Buffer Logic:
    The tracker uses a 2-element deque per worker to create transitions:
    - Single non-terminal step: buffered, no output
    - Two non-terminal steps: emit (prev, prev_result, curr_result), keep current
    - Single terminal step: emit (step, result, None), clear buffer
    - Two steps, second terminal: emit both transition and terminal, clear buffer
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
import numpy as np

from corl.api.wrapper import Wrapper

logger = logging.getLogger(__name__)

REFRESH_RATE = 5  # seconds between API refreshes to update worker list


@dataclass
class StepKey:
    """
    Unique identifier for a specific step in the training process.

    Attributes:
        worker_id: Unique identifier for the worker process
        episode_id: Current episode number for this worker
        step: Step number within the current episode
    """

    worker_id: int
    episode_id: int
    step: int


@dataclass
class StepResult:
    """
    Result data from a single environment step.

    Contains the observation, action taken, reward received, and terminal flag
    for a single timestep in an episode.

    Attributes:
        observation: Environment observation (sensor data, state)
        action: Action taken by the agent (discrete index or continuous values)
        reward: Scalar reward received from the environment
        done: Whether this step terminates the episode
    """

    observation: np.ndarray
    action: int
    reward: float
    done: bool


class Tracker:
    """
    Manages state tracking and transition buffering for distributed RL workers.

    The Tracker coordinates multiple workers executing episodes in parallel. It maintains
    each worker's current position (episode, step) and buffers their step results to
    construct state transitions for training.

    Attributes:
        train_id: Unique identifier for this training session
        api: Wrapper instance for communicating with the training API
        _workers: Mapping of worker_id to current StepKey (position tracker)
        _buffer: Mapping of worker_id to deque of recent (StepKey, StepResult) pairs
        last_refresh: Timestamp of last worker list refresh from API
    """

    train_id: str
    api: Wrapper
    _workers: dict[int, StepKey]
    _buffer: dict[int, deque[tuple[StepKey, StepResult]]]

    def __init__(self, train_id: str, api: Wrapper):
        """
        Initialize a new Tracker for a training session.

        Args:
            train_id: Unique identifier for the training session
            api: Wrapper instance for API communication
        """
        self.train_id = train_id
        self.api = api
        self._workers = {}
        self._buffer = {}
        self.last_refresh = 0
        logger.info(f"Initialized Tracker for training session: {train_id}")

    def workers(self) -> list[StepKey]:
        """
        Get list of all currently tracked workers.

        Returns:
            List of StepKey objects representing current state of each worker
        """
        return list(self._workers.values())

    def refresh(self) -> None:
        """
        Refresh worker list from API and update internal state.

        Queries the API for current worker status (rate-limited to REFRESH_RATE seconds).
        Adds newly active workers, removes inactive workers, and preserves state of
        existing active workers.

        Workers are added with initial state (episode_id=0, step=0) and an empty buffer.
        Workers marked as inactive (status=False) are removed along with their buffers.

        Note:
            This method is rate-limited and will only query the API if REFRESH_RATE
            seconds have elapsed since the last refresh.
        """
        if time.time() - self.last_refresh > REFRESH_RATE:
            workers = self.api.get_workers(self.train_id)
            logger.debug(
                f"Refreshing workers for training {self.train_id}: {len(workers)} workers found"
            )

            for worker in workers:
                worker_id = int(worker["id"])
                status = bool(worker["status"])

                if status and worker_id not in self._workers:
                    self._workers[worker_id] = StepKey(worker_id, 0, 0)
                    self._buffer[worker_id] = deque(maxlen=2)
                    logger.info(f"Added worker {worker_id} to tracker")
                elif not status and worker_id in self._workers:
                    del self._workers[worker_id]
                    del self._buffer[worker_id]
                    logger.info(f"Removed worker {worker_id} from tracker")
                    continue

            self.last_refresh = time.time()

    def increment_step(self, worker_id: int, episode_id: int) -> None:
        """
        Increment the step counter for a worker within the current episode.

        Args:
            worker_id: ID of the worker to increment
            episode_id: Expected current episode ID (for validation)

        Raises:
            ValueError: If worker_id not found or episode_id doesn't match current episode
        """
        self._validate_worker_id(worker_id)
        self._validate_episode_id(worker_id, episode_id)
        self._workers[worker_id].step += 1
        logger.debug(
            f"Incremented step for worker {worker_id} to {self._workers[worker_id].step}"
        )

    def increment_episode(self, worker_id: int) -> None:
        """
        Increment the episode counter for a worker and reset step to 0.

        Called when a worker completes an episode and starts a new one.

        Args:
            worker_id: ID of the worker to increment

        Raises:
            ValueError: If worker_id not found
        """
        self._validate_worker_id(worker_id)
        self._workers[worker_id].episode_id += 1
        self._workers[worker_id].step = 0
        logger.info(
            f"Incremented episode for worker {worker_id} to episode {self._workers[worker_id].episode_id}"
        )

    def add_step_result(
        self,
        step_key: StepKey,
        step_result: StepResult,
    ) -> None:
        """
        Add a step result to the worker's buffer for transition construction.

        Validates that the step matches the worker's current state and appends the
        result to the worker's buffer. The buffer has maxlen=2, so older results
        are automatically discarded when full.

        Args:
            step_key: Identifies the worker, episode, and step for this result
            step_result: Contains observation, action, reward, and done flag

        Raises:
            ValueError: If worker_id not found, episode_id doesn't match, or
                       step doesn't match the worker's current step
        """
        self._validate_worker_id(step_key.worker_id)
        self._validate_episode_id(step_key.worker_id, step_key.episode_id)
        self._validate_step(step_key.worker_id, step_key.step)
        self._buffer[step_key.worker_id].append((step_key, step_result))
        logger.debug(
            f"Added step result for worker {step_key.worker_id}, episode {step_key.episode_id}, step {step_key.step}"
        )

    def get_step_result(self) -> list[tuple[StepKey, StepResult, StepResult | None]]:
        """
        Retrieve available transitions from all worker buffers.

        Processes each worker's buffer to extract complete transitions. The logic varies
        based on buffer state and terminal flags:

        - Empty buffer: No output
        - Single non-terminal step: No output (wait for next step)
        - Single terminal step: Output (step, result, None), clear buffer
        - Two non-terminal steps: Output (prev, prev_result, curr_result), keep current
        - Two steps, second terminal: Output both transition and terminal, clear buffer

        Returns:
            List of tuples, each containing:
                - StepKey: Identifies the step
                - StepResult: Result for this step
                - StepResult | None: Result for next step (None if terminal)

        Note:
            This method modifies buffer state by removing consumed transitions and
            clearing buffers when terminal steps are processed.
        """
        results = []

        for worker_id, buffer in self._buffer.items():
            if not buffer:
                continue

            if len(buffer) == 1:
                step_key, step_result = buffer[0]
                if step_result.done:
                    results.append((step_key, step_result, None))
                    buffer.clear()
                    logger.debug(
                        f"Retrieved terminal step result for worker {worker_id}"
                    )

            elif len(buffer) == 2:
                previous_key, previous_result = buffer[0]
                current_key, current_result = buffer[1]

                if current_result.done:
                    results.append((previous_key, previous_result, current_result))
                    results.append((current_key, current_result, None))
                    buffer.clear()
                    logger.debug(
                        f"Retrieved final two step results for worker {worker_id}"
                    )
                else:
                    results.append((previous_key, previous_result, current_result))
                    buffer.popleft()
                    logger.debug(f"Retrieved step result pair for worker {worker_id}")

        if results:
            logger.debug(f"Retrieved {len(results)} step results from buffer")
        return results

    def _validate_worker_id(self, worker_id: int) -> None:
        """
        Validate that a worker_id exists in the tracker.

        Args:
            worker_id: Worker ID to validate

        Raises:
            ValueError: If worker_id not found in tracker
        """
        if worker_id not in self._workers.keys():
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

    def _validate_step(self, worker_id: int, step: int) -> None:
        """
        Validate that a step number matches the worker's current step.

        Args:
            worker_id: Worker ID to check
            step: Expected step number

        Raises:
            ValueError: If step doesn't match worker's current step
        """
        if step != self._workers[worker_id].step:
            logger.error(
                f"Step mismatch for worker {worker_id}: expected {self._workers[worker_id].step}, got {step}"
            )
            raise ValueError(
                f"Step mismatch for worker {worker_id}: expected {self._workers[worker_id].step}, got {step}."
            )
