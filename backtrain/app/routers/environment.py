"""Environment State Communication Router Message Exchange.

This module provides FastAPI endpoints for publishing and consuming environment state
messages in a distributed reinforcement learning system.

Endpoints:
    - GET  /health                                      : Memory health monitoring
    - POST /{train_id}/{worker_id}/{episode_id}/{step}     : Publish env state
    - GET  /{train_id}/{worker_id}/{episode_id}/{step}     : Consume env state

Key Structure:
    All endpoints use a 4-level hierarchical key for message routing:
    - train_id (str): Training session/experiment identifier
    - worker_id (int): Parallel running instance (0-indexed)
    - episode_id (int): Episode number within training session
    - step (int): Time step within episode

Error Handling:
    - 404: Environment state not found (not yet published or evicted)
    - 422: Invalid request payload (schema validation failed)
    - 500: Memory operation failed
"""

from app.core.memory import Memory
from app.dependencies import get_environment_memory
from app.schemas import EnvironmentSchema, SuccessResponseSchema
from app.schemas.health import MemoryStatsSchema
from fastapi import APIRouter, Depends, HTTPException, Path

router = APIRouter(prefix="/environment", tags=["environment"])


@router.get(
    "/health",
    summary="Monitor environment memory health",
    description="Retrieve real-time memory usage statistics for the environment communication channel. "
    "Provides metrics for monitoring system health, detecting bottlenecks, and capacity planning.",
    response_description="Memory statistics including size, capacity, remaining slots, and usage percentage",
    response_model=MemoryStatsSchema,
    responses={
        200: {
            "description": "Memory health statistics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "size": 3842,
                        "capacity": 10000,
                        "remaining": 6158,
                        "usage_percent": 38.42,
                    }
                }
            },
        }
    },
)
def check_environment_memory_health(
    memory: Memory = Depends(get_environment_memory),
) -> MemoryStatsSchema:
    """Retrieve environment memory health statistics.

    Provides real-time metrics about the environment communication channel's current
    state, including memory usage, capacity, and availability. This endpoint is
    useful for monitoring system health, detecting potential bottlenecks, and
    capacity planning.

    Args:
        memory: Environment memory instance injected via dependency injection.
            Manages the storage and retrieval of environment state messages.

    Returns:
        MemoryStatsSchema: Memory statistics containing:
            - size (int): Current number of stored environment states
            - capacity (int): Maximum number of environment messages that can be stored
            - remaining (int): Available slots before eviction occurs (capacity - size)
            - usage_percent (float): Percentage of capacity currently used (0.0 to 100.0)
    """
    return memory.stats()


@router.post(
    "/{train_id}/{worker_id}/{episode_id}/{step}",
    summary="Publish environment state message",
    description="Environment publishes state (reward, done, data) after action execution. "
    "Trainer consumes this state for decision-making and learning.",
    response_description="Confirmation that environment state was stored successfully",
    status_code=200,
    responses={
        200: {
            "description": "Environment state stored successfully",
            "content": {"application/json": {"example": {"status": "success"}}},
        },
        422: {
            "description": "Validation error - invalid environment schema or path parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "reward"],
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
@router.post("/{train_id}/{worker_id}/{episode_id}/{step}")
def add_environment_step(
    train_id: str = Path(
        ...,
        min_length=1,
        description="Training session identifier (e.g., 'exp_001', 'dqn_cartpole_2024')",
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
    payload: EnvironmentSchema = ...,
    memory: Memory = Depends(get_environment_memory),
) -> SuccessResponseSchema:
    """Publish environment state after action execution.

    This endpoint enables environment to communicate the results of action
    execution to trainer.

    Args:
        train_id: Unique identifier for the training session/experiment.
        worker_id: Running instance number (for parallel workers).
        episode_id: Episode number within the training session.
        step: Time step within the current episode.
        payload: Environment state data conforming to EnvironmentSchema.
        memory: Injected environment memory instance.

    Returns:
        SuccessResponseSchema: Confirmation that environment state was stored successfully.

    """
    key = (train_id, worker_id, episode_id, step)
    memory.add(key, payload)
    return SuccessResponseSchema()


@router.get(
    "/{train_id}/{worker_id}/{episode_id}/{step}",
    summary="Consume environment state message",
    description="Trainer retrieves state published by environment. "
    "Returns 404 if state not yet published or has been evicted from memory.",
    response_description="Environment state data including reward, terminal flag and additional data",
    response_model=EnvironmentSchema,
    responses={
        200: {
            "description": "Environment state retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "done": False,
                        "reward": 1.5,
                        "data": {"position": [0.5, 1.2], "velocity": [0.1, -0.05]},
                    }
                }
            },
        },
        404: {
            "description": "Environment state not found - not yet published or evicted from memory",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Environment not found for train_id=exp_001, worker_id=0, episode_id=5, step=10"
                    }
                }
            },
        },
    },
)
def get_environment_step(
    train_id: str = Path(
        ...,
        min_length=1,
        description="Training session identifier",
    ),
    worker_id: int = Path(
        ...,
        ge=0,
        description="Running instance ID",
    ),
    episode_id: int = Path(
        ...,
        ge=0,
        description="Episode number",
    ),
    step: int = Path(
        ...,
        ge=0,
        description="Time step within episode",
    ),
    memory: Memory = Depends(get_environment_memory),
) -> EnvironmentSchema:
    """Retrieve environment state for trainer learning.

    This endpoint enables trainer consume environment states published by environment.

    Args:
        train_id: Training session identifier.
        worker_id: Running instance number.
        episode_id: Episode number.
        step: Time step to retrieve state for.
        memory: Injected environment memory instance.

    Returns:
        EnvironmentSchema: Environment state data containing:
            - done (bool): Whether episode terminated
            - reward (float): Reward from action execution
            - data (dict): State data with observation and info

    Raises:
        HTTPException: 404 if environment state not found. Reasons:
            - Environment hasn't published state yet (timing issue)
            - State was evicted due to memory capacity limits
            - Invalid key (wrong train_id/worker_id/episode_id/step)
            - Environment brick crashed before publishing
    """
    key = (train_id, worker_id, episode_id, step)
    value = memory.get(key)

    if value is None:
        raise HTTPException(
            status_code=404,
            detail=f"Environment not found for train_id={train_id}, worker_id={worker_id}, episode_id={episode_id}, step={step}",
        )

    return value
