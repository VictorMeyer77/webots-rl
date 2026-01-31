"""Observation Communication Router for Message Exchange.

This module provides FastAPI endpoints for publishing and consuming observation
messages in a distributed reinforcement learning system.
Observations represent data captured by the agent to choose actions.

Error Handling:
    - 404 (GET): Observation not found
        * Observation hasn't been published yet
        * Observation evicted from memory
        * Invalid key (train_id, env_id, episode_id, or step doesn't exist)

    - 422 (POST): Invalid observation data
        * Data not JSON-serializable
        * Schema validation failed (missing required fields)
"""

from app.core.memory import Memory
from app.dependencies import get_observation_memory
from app.schemas import ObservationSchema
from fastapi import APIRouter, Depends, HTTPException, Path

router = APIRouter(prefix="/observation", tags=["observation"])


@router.post(
    "/{train_id}/{env_id}/{episode_id}/{step}",
    summary="Publish observation to memory",
    description=("Agent publishes observation for consumption by trainer."),
    response_description="Confirmation that observation was stored successfully",
    status_code=200,
    responses={
        200: {
            "description": "Observation stored successfully",
            "content": {"application/json": {"example": {"status": "stored"}}},
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
    env_id: int = Path(
        ...,
        ge=0,
        description="Environment instance ID for parallel environments (0-indexed)",
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
) -> dict:
    """Publish observation state to memory.

    The observation is stored in memory
    and made available for agent bricks (decision-making) and trainer bricks
    (experience tuple construction).

    Args:
        train_id: Unique identifier for the training session/experiment.
        env_id: Environment instance number (for parallel environment execution).
        episode_id: Episode number within the training session.
        step: Time step within the current episode.
        payload: ObservationSchema containing observation data and optional metadata.
        memory: Injected observation memory instance (dependency injection).

    Returns:
        dict: Confirmation message with status "stored".
    """
    key = (train_id, env_id, episode_id, step)
    memory.add(key, payload)
    return {"status": "stored"}


@router.get(
    "/{train_id}/{env_id}/{episode_id}/{step}",
    summary="Retrieve observation from memory",
    description=("Trainer retrieve observation state from memory."),
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
                        "detail": "Observation not found for train_id=exp_001, env_id=0, episode_id=5, step=10"
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
    env_id: int = Path(
        ...,
        ge=0,
        description="Environment instance ID for parallel environments (0-indexed)",
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
        env_id: Environment instance number (for parallel environment execution).
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
    key = (train_id, env_id, episode_id, step)
    value = memory.get(key)

    if value is None:
        raise HTTPException(
            status_code=404,
            detail=f"Observation not found for train_id={train_id}, env_id={env_id}, episode_id={episode_id}, step={step}",
        )

    return value
