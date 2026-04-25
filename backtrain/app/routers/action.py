"""Action Communication Router.

This module provides FastAPI endpoints for publishing and consuming action messages
between agent and trainer in a distributed reinforcement learning
system. Actions represent decisions made by trainers that need to be communicated to
agents for execution.

Endpoints:
    - GET  /health                                     : Memory health monitoring
    - POST /{train_id}/{worker_id}/{episode_id}/{step} : Publish single action
    - GET  /{train_id}/{worker_id}/{episode_id}/{step} : Consume single action
    - POST /batch                                      : Retrieve multiple actions (batch GET)
    - POST /batch/publish                              : Publish multiple actions (batch POST)

Key Structure:
    All endpoints use a 4-level hierarchical key for message routing:
    - train_id (str): Training session/experiment identifier
    - worker_id (int): Parallel running instance (0-indexed)
    - episode_id (int): Episode number within training session
    - step (int): Time step within episode

Error Handling:
    - 404: Action not found (not yet published or evicted from memory)
    - 422: Invalid request payload (schema validation failed)
    - 500: Memory operation failed
"""

from app.core.memory import Memory
from app.dependencies import get_action_memory
from app.schemas import ActionSchema, SuccessResponseSchema
from app.schemas.batch import (
    BatchGetRequestSchema,
    BatchGetResponseSchema,
    BatchItemSchema,
    BatchPostRequestSchema,
    BatchPostResponseSchema,
)
from app.schemas.health import MemoryStatsSchema
from fastapi import APIRouter, Depends, HTTPException, Path

router = APIRouter(prefix="/action", tags=["action"])


@router.get(
    "/health",
    summary="Monitor action memory health",
    description="Retrieve real-time memory usage statistics for the action communication channel. "
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
def check_action_memory_health(
    memory: Memory = Depends(get_action_memory),
) -> MemoryStatsSchema:
    """Retrieve action memory health statistics.

    Provides real-time metrics about the action communication channel's current
    state, including memory usage, capacity, and availability. This endpoint is
    useful for monitoring system health, detecting potential bottlenecks, and
    capacity planning.

    Args:
        memory: Action memory instance injected via dependency injection.
            Manages the storage and retrieval of action messages.

    Returns:
        MemoryStatsSchema: Memory statistics containing:
            - size (int): Current number of stored action messages
            - capacity (int): Maximum number of actions that can be stored
            - remaining (int): Available slots before eviction occurs (capacity - size)
            - usage_percent (float): Percentage of capacity currently used (0.0 to 100.0)
    """
    return memory.stats()


@router.post(
    "/{train_id}/{worker_id}/{episode_id}/{step}",
    summary="Publish action message",
    description="Trainer publishes an action for the agent to consume. "
    "The action is stored with a hierarchical key for organized retrieval.",
    response_description="Confirmation that action was stored successfully",
    status_code=200,
    responses={
        200: {
            "description": "Action stored successfully",
            "content": {"application/json": {"example": {"status": "success"}}},
        },
        422: {
            "description": "Validation error - invalid action schema or path parameters"
        },
    },
)
def add_action_step(
    train_id: str = Path(
        ..., min_length=1, description="The training session identifier"
    ),
    worker_id: int = Path(..., ge=0, description="The running instance identifier"),
    episode_id: int = Path(..., ge=0, description="The episode identifier"),
    step: int = Path(..., ge=0, description="The step identifier within the episode"),
    payload: ActionSchema = ...,
    memory: Memory = Depends(get_action_memory),
) -> SuccessResponseSchema:
    """Publish an action.

    This endpoint enables asynchronous communication to publish actions.

    Args:
        train_id: Unique identifier for the training session/experiment.
        worker_id: Running instance number (for parallel workers).
        episode_id: Episode number within the training session.
        step: Time step within the current episode.
        payload: Action data conforming to ActionSchema.
        memory: Injected action memory instance.

    Returns:
        SuccessResponseSchema: Confirmation that the action was stored successfully.

    """
    key = (train_id, worker_id, episode_id, step)
    memory.add(key, payload)
    return SuccessResponseSchema()


@router.get(
    "/{train_id}/{worker_id}/{episode_id}/{step}",
    summary="Consume action message",
    description="Agent retrieves an action published by the trainer. "
    "Returns 404 if action not yet published or has been evicted.",
    response_description="Action data for environment execution",
    response_model=ActionSchema,
    responses={
        200: {
            "description": "Action retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "action": 2,
                    }
                }
            },
        },
        404: {
            "description": "Action not found - not yet published or evicted from memory",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Action not found for train_id=exp_001, worker_id=0, episode_id=5, step=10"
                    }
                }
            },
        },
    },
)
def get_action_step(
    train_id: str = Path(
        ..., min_length=1, description="The training session identifier"
    ),
    worker_id: int = Path(..., ge=0, description="The running instance identifier"),
    episode_id: int = Path(..., ge=0, description="The episode identifier"),
    step: int = Path(..., ge=0, description="The step identifier within the episode"),
    memory: Memory = Depends(get_action_memory),
) -> ActionSchema:
    """Retrieve an action.

    This endpoint enables agents to consume actions published by trainers.

    Args:
        train_id: Training session identifier.
        worker_id: Running instance number.
        episode_id: Episode number.
        step: Time step to retrieve action for.
        memory: Injected action memory instance.

    Returns:
        ActionSchema: Action data containing:
            - action: The action value(s) to execute

    Raises:
        HTTPException: 404 if action not found.
    """
    key = (train_id, worker_id, episode_id, step)
    value = memory.get(key)

    if value is None:
        raise HTTPException(
            status_code=404,
            detail=f"Action not found for train_id={train_id}, worker_id={worker_id}, episode_id={episode_id}, step={step}",
        )

    return value


@router.post(
    "/batch",
    summary="Retrieve multiple actions in batch",
    description="Retrieve multiple actions in a single request by providing a list of keys. "
    "This endpoint is more efficient than making multiple individual GET requests. "
    "Returns results for all requested keys, indicating which were found and which were not.",
    response_description="Batch results containing actions and retrieval status for each key",
    response_model=BatchGetResponseSchema,
    responses={
        200: {
            "description": "Batch retrieval completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "results": [
                            {
                                "key": {
                                    "train_id": "exp_001",
                                    "worker_id": 0,
                                    "episode_id": 5,
                                    "step": 10,
                                },
                                "value": {"action": 2, "executed": False},
                                "found": True,
                            },
                            {
                                "key": {
                                    "train_id": "exp_001",
                                    "worker_id": 0,
                                    "episode_id": 5,
                                    "step": 11,
                                },
                                "value": None,
                                "found": False,
                            },
                        ],
                        "total": 2,
                        "found": 1,
                        "missing": 1,
                    }
                }
            },
        },
        422: {
            "description": "Validation error - invalid request payload",
        },
    },
)
def get_action_batch(
    payload: BatchGetRequestSchema,
    memory: Memory = Depends(get_action_memory),
) -> BatchGetResponseSchema:
    """Retrieve multiple actions in a single batch request.

    This endpoint provides efficient retrieval of multiple actions by accepting
    a list of keys and returning all corresponding actions in a single response.
    Unlike individual GET requests, this endpoint does not raise errors for
    missing actions - instead, it indicates in the response which actions were
    found and which were not.

    Args:
        payload: Batch request containing a list of action keys to retrieve.
        memory: Injected action memory instance.

    Returns:
        BatchGetResponseSchema: Batch response containing:
            - results: List of action results with keys, actions, and found status
            - total: Total number of keys requested
            - found: Count of successfully retrieved actions
            - missing: Count of actions not found in memory
    """
    results = []
    found_count = 0
    missing_count = 0

    for key_schema in payload.keys:
        key = (
            key_schema.train_id,
            key_schema.worker_id,
            key_schema.episode_id,
            key_schema.step,
        )
        value = memory.get(key)

        results.append(BatchItemSchema(key=key_schema, value=value))
        found_count += 1 if value is not None else 0
        missing_count += 1 if value is None else 0

    return BatchGetResponseSchema(
        results=results,
        total=len(payload.keys),
        found=found_count,
        missing=missing_count,
    )


@router.post(
    "/batch/publish",
    summary="Publish multiple actions in batch",
    description="Publish multiple actions in a single request. "
    "This endpoint is more efficient than making multiple individual POST requests. "
    "All actions are stored in memory with their respective keys.",
    response_description="Confirmation of batch publish operation with counts",
    response_model=BatchPostResponseSchema,
    responses={
        200: {
            "description": "Batch publish completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "total": 10,
                        "published": 10,
                    }
                }
            },
        },
        422: {
            "description": "Validation error - invalid request payload",
        },
    },
)
def add_action_batch(
    payload: BatchPostRequestSchema,
    memory: Memory = Depends(get_action_memory),
) -> BatchPostResponseSchema:
    """Publish multiple actions in a single batch request.

    This endpoint provides efficient publishing of multiple actions by accepting
    a list of action items (key + action data) and storing all of them in memory.
    This is useful for bulk operations, reducing network overhead and improving
    throughput when publishing many actions at once.

    Args:
        payload: Batch publish request containing a list of items (key-value pairs) to store.
        memory: Injected action memory instance.

    Returns:
        BatchPostResponseSchema: Batch response containing:
            - status: Success status message
            - total: Total number of items in the request
    """

    for item in payload.items:
        key = (
            item.key.train_id,
            item.key.worker_id,
            item.key.episode_id,
            item.key.step,
        )
        memory.add(key, item.value)

    return BatchPostResponseSchema(
        total=len(payload.items),
    )
