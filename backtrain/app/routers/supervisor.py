"""FastAPI router for managing training session supervision.

This module provides REST API endpoints for coordinating multiple parallel
workers during reinforcement learning training sessions. It handles:

- Creating and managing training sessions
- Registering workers for training sessions
- Tracking episode progress across workers
- Synchronizing episode counters

The supervisor ensures that parallel workers can operate independently while
maintaining consistent state through a centralized coordination service.

Typical workflow:
    1. Create a training session with POST /supervisor/train
    2. Register workers with POST /supervisor/train/{train_id}/worker
    3. Workers query their episode ID with GET /supervisor/train/{train_id}/worker/{worker_id}/episode
    4. Workers increment their episode counter with POST /supervisor/train/{train_id}/worker/{worker_id}/episode/increment
    5. Optionally, update worker status with POST /supervisor/train/{train_id}/worker/{worker_id}/status

All endpoints return appropriate HTTP status codes and error messages for
invalid operations (e.g., duplicate training sessions, non-existent workers).
"""

from app.core.supervisor import Supervisor, SupervisorError
from app.dependencies import get_supervisor
from app.schemas import SuccessResponseSchema
from app.schemas.supervisor import (
    EpisodeIdSchema,
    TrainIdSchema,
    TrainingSchema,
    WorkerIdSchema,
)
from fastapi import APIRouter, Depends, HTTPException, status

router = APIRouter(prefix="/supervisor", tags=["supervisor"])


@router.post(
    "/train",
    status_code=status.HTTP_201_CREATED,
    summary="Create a new training session",
    description="Register a new training session with a unique identifier. "
    "This session can then have multiple workers assigned to it.",
    responses={
        201: {
            "description": "Training session created successfully",
            "content": {"application/json": {"example": {"status": "success"}}},
        },
        409: {
            "description": "Training session with this ID already exists",
            "content": {
                "application/json": {
                    "example": {"detail": "Train ID train_001 already exists."}
                }
            },
        },
    },
)
def add_train(
    request: TrainIdSchema, supervisor: Supervisor = Depends(get_supervisor)
) -> SuccessResponseSchema:
    """Register a new training session.

    Args:
        request: Request body containing the training session ID.
        supervisor: Supervisor instance injected as dependency.

    Returns:
        Success confirmation message.

    Raises:
        HTTPException: 409 if the training session already exists.
    """
    try:
        supervisor.add_train(request.train_id)
        return SuccessResponseSchema()
    except SupervisorError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get(
    "/train/{train_id}",
    summary="Get training session details",
    description="Retrieve detailed information about a specific training session, "
    "including all registered workers, their episode IDs, and active status. "
    "This endpoint provides a complete snapshot of the current training state.",
    responses={
        200: {
            "description": "Training session details retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "train_001",
                        "workers": [
                            {"id": 0, "episode_id": 42, "status": True},
                            {"id": 1, "episode_id": 38, "status": False},
                        ],
                    }
                }
            },
        },
        404: {
            "description": "Training session not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Train ID train_001 does not exist."}
                }
            },
        },
    },
)
def get_train(
    train_id: str, supervisor: Supervisor = Depends(get_supervisor)
) -> TrainingSchema:
    try:
        return supervisor.get_train(train_id)
    except SupervisorError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post(
    "/train/{train_id}/worker",
    status_code=status.HTTP_201_CREATED,
    summary="Add a worker to a training session",
    description="Register a new worker for an existing training session. "
    "The worker is automatically assigned a unique ID and starts at episode 0.",
    responses={
        201: {
            "description": "Worker created successfully",
            "content": {"application/json": {"example": {"worker_id": 0}}},
        },
        404: {
            "description": "Training session not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Train ID train_001 does not exist."}
                }
            },
        },
    },
)
def add_worker(
    train_id: str, supervisor: Supervisor = Depends(get_supervisor)
) -> WorkerIdSchema:
    """Register a new worker for an existing training session.

    Args:
        train_id: Training session identifier.
        supervisor: Supervisor instance injected as dependency.

    Returns:
        The newly assigned worker ID.

    Raises:
        HTTPException: 404 if the training session doesn't exist.
    """
    try:
        worker_id = supervisor.add_worker(train_id)
        return WorkerIdSchema(worker_id=worker_id)
    except SupervisorError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post(
    "/train/{train_id}/worker/{worker_id}/status",
    summary="Update worker status",
    description="Update the active status flag for a specific worker in a training session. "
    "This allows marking workers as active or inactive during training without "
    "affecting their episode counter.",
    responses={
        200: {
            "description": "Worker status updated successfully",
            "content": {"application/json": {"example": {"status": "success"}}},
        },
        404: {
            "description": "Training session or worker not found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Worker ID 5 does not exist for Train ID train_001."
                    }
                }
            },
        },
    },
)
def update_worker_status(
    train_id: str,
    worker_id: int,
    worker_status: bool,
    supervisor: Supervisor = Depends(get_supervisor),
) -> SuccessResponseSchema:
    """Update the active status of a specific worker.

    Args:
        train_id: Training session identifier.
        worker_id: Worker identifier.
        worker_status: New active status for the worker (True for active, False for inactive).
        supervisor: Supervisor instance injected as dependency.

    Returns:
        Success confirmation message.

    Raises:
        HTTPException: 404 if the training session or worker doesn't exist.
    """
    try:
        supervisor.update_worker_status(train_id, worker_id, worker_status)
        return SuccessResponseSchema()
    except SupervisorError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get(
    "/train/{train_id}/worker/{worker_id}/episode",
    summary="Get current episode ID for a worker",
    description="Retrieve the current episode ID that a specific worker is on. "
    "This is useful for tracking training progress across parallel workers.",
    responses={
        200: {
            "description": "Episode ID retrieved successfully",
            "content": {"application/json": {"example": {"episode_id": 42}}},
        },
        404: {
            "description": "Training session or worker not found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Worker ID 5 does not exist in train train_001."
                    }
                }
            },
        },
    },
)
def get_episode_id(
    train_id: str, worker_id: int, supervisor: Supervisor = Depends(get_supervisor)
) -> EpisodeIdSchema:
    """Retrieve the current episode ID for a specific worker.

    Args:
        train_id: Training session identifier.
        worker_id: Worker identifier.
        supervisor: Supervisor instance injected as dependency.

    Returns:
        The current episode ID for the specified worker.

    Raises:
        HTTPException: 404 if the training session or worker doesn't exist.
    """
    try:
        episode_id = supervisor.get_episode_id(train_id, worker_id)
        return EpisodeIdSchema(episode_id=episode_id)
    except SupervisorError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post(
    "/train/{train_id}/worker/{worker_id}/episode/increment",
    summary="Increment episode ID for a worker",
    description="Increment the episode counter for a specific worker by 1. "
    "Call this endpoint when a worker completes an episode.",
    responses={
        200: {
            "description": "Episode ID incremented successfully",
            "content": {"application/json": {"example": {"status": "success"}}},
        },
        404: {
            "description": "Training session or worker not found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Worker ID 5 does not exist in train train_001."
                    }
                }
            },
        },
    },
)
def increment_episode_id(
    train_id: str, worker_id: int, supervisor: Supervisor = Depends(get_supervisor)
) -> SuccessResponseSchema:
    """Increment the episode ID for a specific worker.

    Args:
        train_id: Training session identifier.
        worker_id: Worker identifier.
        supervisor: Supervisor instance injected as dependency.

    Returns:
        Success confirmation message.

    Raises:
        HTTPException: 404 if the training session or worker doesn't exist.
    """
    try:
        supervisor.increment_episode_id(train_id, worker_id)
        return SuccessResponseSchema()
    except SupervisorError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
