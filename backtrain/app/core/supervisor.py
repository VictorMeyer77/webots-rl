"""Core supervisor module for managing distributed reinforcement learning training sessions.

This module provides the `Supervisor` class which coordinates multiple parallel workers
across different training sessions. It maintains state for worker assignments and tracks
episode progress for each worker independently.

The supervisor uses an in-memory dictionary structure to store training session data:
    - Training sessions are identified by string IDs (e.g., "train_001")
    - Each training session contains multiple workers identified by integer IDs
    - Each worker tracks its current episode number as an integer

Classes:
    SupervisorError: Custom exception for supervisor-related errors.
    Supervisor: Main coordinator for training sessions and workers.
"""


class SupervisorError(Exception):
    """Custom exception raised when supervisor operations fail.

    This exception is raised for various error conditions including:
    - Attempting to create a training session with a duplicate ID
    - Attempting to create a training session with an empty ID
    - Attempting to access a non-existent training session
    - Attempting to access a non-existent worker

    Attributes:
        Inherits all attributes from the base Exception class.
    """

    pass


class Supervisor:
    """Coordinator for distributed reinforcement learning training sessions.

    The Supervisor maintains state for multiple training sessions running in parallel,
    where each session can have multiple workers processing episodes independently.
    Workers are assigned sequential IDs starting from 0 and track their progress
    through incrementing episode counters.

    Thread Safety:
        This implementation is NOT thread-safe. For concurrent access, external
        synchronization mechanisms must be used.
    """

    _training: dict[str, dict[int, int]]

    def __init__(self):
        """Initialize a new Supervisor with empty training session storage.

        Creates an empty dictionary to store all training sessions and their
        associated worker states.
        """
        self._training: dict[str, dict[int, int]] = {}

    @property
    def training(self) -> dict[str, dict[int, int]]:
        """Get the current training sessions and worker states.

        Returns:
            Dictionary mapping training session IDs to dictionaries of worker states.
            Each worker state maps worker ID to current episode ID.
        """
        return self._training

    def add_train(self, train_id: str):
        """Register a new training session.

        Creates a new training session with the specified ID. The session starts
        with no workers and must have a unique, non-empty identifier.

        Args:
            train_id: Unique identifier for the training session. Must be non-empty.

        Raises:
            SupervisorError: If train_id is empty or already exists.
        """
        if train_id == "":
            raise SupervisorError("Train ID cannot be empty.")
        if train_id not in self.training:
            self.training[train_id] = {}
        else:
            raise SupervisorError(f"Train ID {train_id} already exists.")

    def add_worker(self, train_id: str) -> int:
        """Register a new worker for an existing training session.

        Creates a new worker with an automatically assigned sequential ID.
        Worker IDs start at 0 and increment for each new worker in the session.
        The worker is initialized with episode_id 0.

        Args:
            train_id: Identifier of the training session to add the worker to.

        Returns:
            The automatically assigned worker ID (non-negative integer).

        Raises:
            SupervisorError: If the training session does not exist.
        """
        if train_id in self.training:
            if len(self.training[train_id]) > 0:
                worker_id = max(self.training[train_id].keys()) + 1
            else:
                worker_id = 0
            self.training[train_id][worker_id] = 0
            return worker_id
        else:
            raise SupervisorError(f"Train ID {train_id} does not exist.")

    def get_episode_id(self, train_id: str, worker_id: int) -> int:
        """Retrieve the current episode ID for a specific worker.

        Returns the episode counter for the specified worker, indicating how many
        episodes the worker has completed.

        Args:
            train_id: Identifier of the training session.
            worker_id: Identifier of the worker within the training session.

        Returns:
            The current episode ID (non-negative integer).

        Raises:
            SupervisorError: If the training session or worker does not exist.
        """
        if train_id in self.training:
            if worker_id in self.training[train_id]:
                return self.training[train_id][worker_id]
            else:
                raise SupervisorError(
                    f"Worker ID {worker_id} does not exist for Train ID {train_id}."
                )
        else:
            raise SupervisorError(f"Train ID {train_id} does not exist.")

    def increment_episode_id(self, train_id: str, worker_id: int) -> None:
        """Increment the episode counter for a specific worker by 1.

        Call this method when a worker completes an episode to track its progress.
        The episode counter increments indefinitely without an upper bound.

        Args:
            train_id: Identifier of the training session.
            worker_id: Identifier of the worker within the training session.

        Returns:
            None

        Raises:
            SupervisorError: If the training session or worker does not exist.
        """
        if train_id in self.training:
            if worker_id in self.training[train_id]:
                self.training[train_id][worker_id] += 1
            else:
                raise SupervisorError(
                    f"Worker ID {worker_id} does not exist for Train ID {train_id}."
                )
        else:
            raise SupervisorError(f"Train ID {train_id} does not exist.")
