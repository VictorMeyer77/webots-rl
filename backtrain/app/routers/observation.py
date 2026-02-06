"""Observation Communication Router for Message Exchange.

This module provides FastAPI endpoints for publishing and consuming observation
messages in a distributed reinforcement learning system.
Observations represent data captured by the agent to choose actions.

Error Handling:
    - 404 (GET): Observation not found
        * Observation hasn't been published yet
        * Observation evicted from memory
        * Invalid key (train_id, worker_id, episode_id, or step doesn't exist)

    - 422 (POST): Invalid observation data
        * Data not JSON-serializable
        * Schema validation failed (missing required fields)
"""

from app.core.memory import Memory
from app.dependencies import get_observation_memory
from app.schemas import ObservationSchema, SuccessResponseSchema
from app.schemas.health import MemoryStatsSchema
from fastapi import APIRouter, Depends, HTTPException, Path

router = APIRouter(prefix="/observation", tags=["observation"])


@router.get(
    "/health",
    summary="Monitor observation memory health",
    description="Retrieve real-time memory usage statistics for the observation communication channel. "
    "Provides metrics for monitoring system health, detecting bottlenecks, and capacity planning.",
    response_description="Memory statistics including size, capacity, remaining slots, and usage percentage",
    response_model=MemoryStatsSchema,
    responses={
        200: {
            "description": "Memory health statistics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "size": 4523,
                        "capacity": 10000,
                        "remaining": 5477,
                        "usage_percent": 45.23,
                    }
                }
            },
        }
    },
)
def check_observation_memory_health(
    memory: Memory = Depends(get_observation_memory),
) -> MemoryStatsSchema:
    """Retrieve observation memory health statistics.

    Provides real-time metrics about the observation communication channel's current
    state, including memory usage, capacity, and availability. This endpoint is
    useful for monitoring system health, detecting potential bottlenecks, and
    capacity planning.

    Args:
        memory: Observation memory instance injected via dependency injection.
            Manages the storage and retrieval of observation messages.

    Returns:
        MemoryStatsSchema: Memory statistics containing:
            - size (int): Current number of stored observation messages
            - capacity (int): Maximum number of observations that can be stored
            - remaining (int): Available slots before eviction occurs (capacity - size)
            - usage_percent (float): Percentage of capacity currently used (0.0 to 100.0)
    """
    return memory.stats()


@router.post(
    "/{train_id}/{worker_id}/{episode_id}/{step}",
    summary="Publish observation to memory",
    description=("Agent publishes observation for consumption by trainer."),
    response_description="Confirmation that observation was stored successfully",
    status_code=200,
    responses={
        200: {
            "description": "Observation stored successfully",
            "content": {"application/json": {"example": {"status": "success"}}},
        },
        422: {
            "description": "Invalid observation data (schema validation failed)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "data"],
                                "msg": "field required",
                                "type": "value_error.missing",
                            }
                        ]
                    }
                }
            },
        },
    },
)
def add_observation_step(
    train_id: str = Path(
        ...,
        min_length=1,
        max_length=100,
        description="Training session identifier",
    ),
    worker_id: int = Path(
        ...,
        ge=0,
        description="Running instance ID for parallel workers (0-indexed)",
    ),
    episode_id: int = Path(
        ...,
        ge=0,
        description="Episode number within the training session",
    ),
    step: int = Path(
        ...,
        ge=0,
        description="Time step within the episode (0-indexed)",
    ),
    payload: ObservationSchema = ...,
    memory: Memory = Depends(get_observation_memory),
) -> SuccessResponseSchema:
    """Publish observation state to memory.

    The observation is stored in memory
    and made available for agent bricks (decision-making) and trainer bricks
    (experience tuple construction).

    Args:
        train_id: Unique identifier for the training session/experiment.
        worker_id: Running instance number (for parallel environment execution).
        episode_id: Episode number within the training session.
        step: Time step within the current episode.
        payload: ObservationSchema containing observation data and optional metadata.
        memory: Injected observation memory instance (dependency injection).

    Returns:
        SuccessResponseSchema: Confirmation of successful storage with status message.
    """
    key = (train_id, worker_id, episode_id, step)
    memory.add(key, payload)
    return SuccessResponseSchema()


@router.get(
    "/{train_id}/{worker_id}/{episode_id}/{step}",
    summary="Retrieve observation from memory",
    description="Trainer retrieve observation state from memory.",
    response_description="Observation data.",
    response_model=ObservationSchema,
    responses={
        200: {
            "description": "Observation retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "data": [0.1, 0.5, 0.2, -0.3],
                    }
                }
            },
        },
        404: {
            "description": "Observation not found (not published yet or evicted from memory)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Observation not found for train_id=exp_001, worker_id=0, episode_id=5, step=10"
                    }
                }
            },
        },
    },
)
def get_observation_step(
    train_id: str = Path(
        ...,
        min_length=1,
        max_length=100,
        description="Training session identifier",
    ),
    worker_id: int = Path(
        ...,
        ge=0,
        description="Running instance ID for parallel workers (0-indexed)",
    ),
    episode_id: int = Path(
        ...,
        ge=0,
        description="Episode number within the training session",
    ),
    step: int = Path(
        ...,
        ge=0,
        description="Time step within the episode (0-indexed)",
    ),
    memory: Memory = Depends(get_observation_memory),
) -> ObservationSchema:
    """Retrieve observation state from memory.

    Args:
        train_id: Unique identifier for the training session/experiment.
        worker_id: Running instance number (for parallel environment execution).
        episode_id: Episode number within the training session.
        step: Time step within the current episode.
        memory: Injected observation memory instance (dependency injection).

    Returns:
        ObservationSchema: Observation data with optional metadata.
            - data: Observation dict

    Raises:
        HTTPException: 404 if observation not found. Reasons:
            - Observation hasn't been published yet
            - Observation evicted from memory
            - Invalid key
    """
    key = (train_id, worker_id, episode_id, step)
    value = memory.get(key)

    if value is None:
        raise HTTPException(
            status_code=404,
            detail=f"Observation not found for train_id={train_id}, worker_id={worker_id}, episode_id={episode_id}, step={step}",
        )

    return value
