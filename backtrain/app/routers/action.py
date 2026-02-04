"""Action Communication Router.

This module provides FastAPI endpoints for publishing and consuming action messages
between agent and trainer in a distributed reinforcement learning
system. Actions represent decisions made by trainers that need to be communicated to
agents for execution.

Endpoints:
    - GET  /health                                  : Memory health monitoring
    - POST /{train_id}/{worker_id}/{episode_id}/{step} : Publish action
    - GET  /{train_id}/{worker_id}/{episode_id}/{step} : Consume action

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
from app.schemas import ActionSchema
from fastapi import APIRouter, Depends, HTTPException, Path

router = APIRouter(prefix="/action", tags=["action"])


@router.get(
    "/health",
    summary="Monitor action memory health",
    description="Retrieve memory usage statistics for monitoring and capacity planning",
    response_description="Memory statistics including size, capacity, and usage percentage",
    responses={
        200: {
            "description": "Memory health statistics",
            "content": {
                "application/json": {
                    "example": {
                        "stats": {
                            "size": 4523,
                            "capacity": 10000,
                            "remaining": 5477,
                            "usage_percent": 45.23,
                        }
                    }
                }
            },
        }
    },
)
@router.get("/health")
def check_action_memory_health(
    memory: Memory = Depends(get_action_memory),
) -> dict:
    """Get action memory health statistics.

    Provides real-time metrics about the action communication channel's current
    state. Useful for monitoring system health and capacity planning.

    Args:
        memory: Injected action memory instance via dependency injection.

    Returns:
        dict: Dictionary containing:
            - stats (dict): Memory statistics with fields:
                - size (int): Current number of stored actions
                - capacity (int): Maximum action capacity
                - remaining (int): Available slots before eviction
                - usage_percent (float): Percentage of capacity used
    """
    stats = memory.stats()
    return {"stats": stats}


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
            "content": {"application/json": {"example": {"status": "stored"}}},
        },
        422: {
            "description": "Validation error - invalid action schema or path parameters"
        },
    },
)
@router.post("/{train_id}/{worker_id}/{episode_id}/{step}")
def add_action_step(
    train_id: str = Path(
        ..., min_length=1, description="The training session identifier"
    ),
    worker_id: int = Path(..., ge=0, description="The running instance identifier"),
    episode_id: int = Path(..., ge=0, description="The episode identifier"),
    step: int = Path(..., ge=0, description="The step identifier within the episode"),
    payload: ActionSchema = ...,
    memory: Memory = Depends(get_action_memory),
) -> dict:
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
        dict: Status confirmation with key:
            - status (str): "stored" if successful

    """
    key = (train_id, worker_id, episode_id, step)
    memory.add(key, payload)
    return {"status": "stored"}


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

    This endpoint enables component to consume actions.

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
