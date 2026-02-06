"""Supervisor-related schemas for training session management.

This module defines Pydantic schemas used for managing training sessions and
workers through the supervisor API. It provides structured data models for
creating sessions, managing workers, and tracking their state.

The schemas are used by supervisor endpoints to handle requests and return
consistent, validated responses about training sessions, workers, and episodes.
"""

from pydantic import BaseModel


class TrainIdSchema(BaseModel):
    """Request model for creating a new training session.

    Used to specify the identifier for a new training session when initiating
    the training process through the supervisor API.

    Attributes:
        train_id: Unique identifier for the training session.
    """

    train_id: str


class WorkerIdSchema(BaseModel):
    """Response model for adding a worker to a training session.

    Returned when a new worker is successfully added to an existing training
    session, providing the assigned worker identifier.

    Attributes:
        worker_id: Unique numerical identifier assigned to the worker.
    """

    worker_id: int


class EpisodeIdSchema(BaseModel):
    """Response model for retrieving an episode ID.

    Returned when querying for the current episode identifier associated with
    a worker in a training session.

    Attributes:
        episode_id: Unique numerical identifier for the training episode.
    """

    episode_id: int


class WorkerSchema(BaseModel):
    """Worker information within a training session.

    Represents a single worker's state and current episode assignment in an
    active training session.

    Attributes:
        id: Unique numerical identifier for the worker.
        episode_id: Current episode the worker is executing.
        status: Worker operational status (True if active, False if inactive).
    """

    id: int
    episode_id: int
    status: bool


class TrainingSchema(BaseModel):
    """Training session information.

    Represents a complete training session including its identifier and all
    associated workers. Used for monitoring active training sessions and their
    worker states.

    Attributes:
        id: Unique identifier for the training session.
        workers: List of all workers participating in this training session.
    """

    id: str
    workers: list[WorkerSchema]
