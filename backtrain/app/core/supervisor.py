"""Supervisor module for managing training sessions and workers.

This module provides the Supervisor class, which tracks multiple training sessions
and their associated workers. Each training session can have multiple workers, and
each worker maintains an episode counter and active status flag.

The module also defines SupervisorError, a custom exception used for supervisor-related
errors such as invalid training IDs or worker IDs.

Classes:
    SupervisorError: Custom exception for supervisor operations.
    Supervisor: Manages training sessions and worker states.
"""

from app.schemas.supervisor import TrainingSchema, WorkerSchema


class SupervisorError(Exception):
    """Exception raised for errors in supervisor operations.

    This exception is raised when invalid operations are performed on training
    sessions or workers, such as accessing non-existent training IDs, duplicate
    training IDs, or invalid worker IDs.
    """

    pass


class Supervisor:
    """Manages training sessions and their associated workers.

    The Supervisor class provides functionality to create and manage multiple training
    sessions, each containing multiple workers. Each worker maintains an episode counter
    and an active status flag. The class supports operations such as adding training
    sessions, registering workers, tracking episode progress, and managing worker status.

    Note:
        This class is not thread-safe. Concurrent access should be synchronized externally.

    Attributes:
        _training: Dictionary mapping training session IDs to their workers.
                  Structure: {train_id: {worker_id: (episode_id, status)}}
    """

    _training: dict[str, dict[int, tuple[int, bool]]]

    def __init__(self):
        """Initialize a new Supervisor with empty training session storage.

        Creates an empty dictionary to store all training sessions and their
        associated worker states.
            The structure is:
            {
                "train_id_1": {
                    1: (episode_id, status),
                    2: (episode_id, status),
                    ...
                },
                "train_id_2": {
                    1: (episode_id, status),
                    ...
                },
                ...
            }
            where:
            - train_id is a unique string identifier for the training session
            - worker_id is a unique integer identifier for the worker within the session
            - episode_id is an integer counter of completed episodes for that worker
            - status is a boolean indicating the worker's active status
        """
        self._training: dict[str, dict[int, tuple[int, bool]]] = {}

    @property
    def trainings(self) -> list[TrainingSchema]:
        """Get all training sessions with their workers in schema format.

        Converts the internal training data structure to a structured schema
        representation suitable for API responses or serialization.

        Returns:
            SupervisorSchema: Schema containing all training sessions, each with
                their list of workers including worker ID, episode counter, and status.
        """
        trainings = []
        for train_id, workers in self._training.items():
            worker_schemas = []
            for worker_id, (episode_id, status) in workers.items():
                worker_schemas.append(
                    WorkerSchema(
                        id=worker_id,
                        episode_id=episode_id,
                        status=status,
                    )
                )
            trainings.append(
                TrainingSchema(
                    id=train_id,
                    workers=worker_schemas,
                )
            )
        return trainings

    def add_train(self, train_id: str):
        """Create a new training session with the specified identifier.

        Initializes an empty training session that can contain multiple workers.
        The training session is created with no workers initially.

        Args:
            train_id: Unique string identifier for the training session.
                     Cannot be empty.

        Raises:
            SupervisorError: If train_id is empty or already exists.
        """
        if train_id == "":
            raise SupervisorError("Train ID cannot be empty.")
        if train_id not in self._training:
            self._training[train_id] = {}
        else:
            raise SupervisorError(f"Train ID {train_id} already exists.")

    def get_train(self, train_id: str) -> TrainingSchema:
        """Retrieve a training session by its identifier.

        Args:
            train_id: Unique string identifier for the training session.

        Returns:
            TrainingSchema: Schema representation of the training session,
                including its ID and list of workers.
        """
        self._validate_train_exists(train_id)
        workers = []
        for worker_id, (episode_id, status) in self._training[train_id].items():
            workers.append(
                WorkerSchema(
                    id=worker_id,
                    episode_id=episode_id,
                    status=status,
                )
            )
        return TrainingSchema(id=train_id, workers=workers)

    def add_worker(self, train_id: str) -> int:
        """Register a new worker for an existing training session.

        Creates a new worker with an auto-incremented ID and initializes its
        episode counter to 0 with active status set to True. Worker IDs start
        at 0 and increment sequentially.

        Args:
            train_id: Identifier of the training session to add the worker to.

        Returns:
            The unique integer ID assigned to the newly created worker.

        Raises:
            SupervisorError: If the training session does not exist.
        """
        self._validate_train_exists(train_id)
        if len(self._training[train_id]) > 0:
            worker_id = max(self._training[train_id].keys()) + 1
        else:
            worker_id = 0
        self._training[train_id][worker_id] = (0, True)
        return worker_id

    def get_worker(self, train_id: str, worker_id: int) -> WorkerSchema:
        """Retrieve a specific worker from a training session.

        Fetches the worker's current state including its ID, episode counter,
        and active status flag.

        Args:
            train_id: Identifier of the training session.
            worker_id: Identifier of the worker within the training session.

        Returns:
            WorkerSchema: Schema representation of the worker, including
                its ID, current episode counter, and active status.

        Raises:
            SupervisorError: If the training session or worker does not exist.
        """
        self._validate_worker_exists(train_id, worker_id)
        return WorkerSchema(
            id=worker_id,
            episode_id=self._training[train_id][worker_id][0],
            status=self._training[train_id][worker_id][1],
        )

    def increment_episode_id(self, train_id: str, worker_id: int) -> None:
        """Increment the episode counter for a specific worker by one.

        This method should be called when a worker completes an episode.
        The worker's status flag is preserved during the increment.

        Args:
            train_id: Identifier of the training session.
            worker_id: Identifier of the worker within the training session.

        Raises:
            SupervisorError: If the training session or worker does not exist.
        """
        self._validate_worker_exists(train_id, worker_id)
        episode_id, status = self._training[train_id][worker_id]
        self._training[train_id][worker_id] = (episode_id + 1, status)

    def update_worker_status(self, train_id: str, worker_id: int, status: bool) -> None:
        """Update the active status flag for a specific worker.

        Modifies the worker's status while preserving its episode counter.
        This can be used to mark workers as active or inactive during training.

        Args:
            train_id: Identifier of the training session.
            worker_id: Identifier of the worker within the training session.
            status: Boolean flag indicating the worker's active status.
                   True for active, False for inactive.

        Raises:
            SupervisorError: If the training session or worker does not exist.
        """
        self._validate_worker_exists(train_id, worker_id)
        episode_id, _ = self._training[train_id][worker_id]
        self._training[train_id][worker_id] = (episode_id, status)

    def _validate_train_exists(self, train_id: str) -> None:
        """Validate that a training session exists.

        Internal helper method to check if a training session ID is registered.

        Args:
            train_id: Identifier of the training session to validate.

        Raises:
            SupervisorError: If the training session does not exist.
        """
        if train_id not in self._training:
            raise SupervisorError(f"Train ID {train_id} does not exist.")

    def _validate_worker_exists(self, train_id: str, worker_id: int) -> None:
        """Validate that a worker exists within a training session.

        Internal helper method that first validates the training session exists,
        then checks if the worker is registered in that session.

        Args:
            train_id: Identifier of the training session.
            worker_id: Identifier of the worker to validate.

        Raises:
            SupervisorError: If the training session or worker does not exist.
        """
        self._validate_train_exists(train_id)
        if worker_id not in self._training[train_id]:
            raise SupervisorError(
                f"Worker ID {worker_id} does not exist for Train ID {train_id}."
            )
