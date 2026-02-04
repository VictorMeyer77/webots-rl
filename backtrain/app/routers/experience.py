"""Experience Tuple Communication Router for Reinforcement Learning Training.

This module provides FastAPI endpoints for retrieving complete experience tuples
by aggregating messages from multiple communication channels (observation, action,
environment).

Experience Tuple Structure:
    An experience tuple represents a complete state transition in RL:
    - observation (s_t): State at time t
    - action (a_t): Action taken at time t
    - reward (r_t): Immediate reward from action execution
    - done: Whether episode terminated at this step
    - next_observation (s_t+1): Resulting state at time t+1
"""

from app.core.memory import Memory
from app.dependencies import (
    get_action_memory,
    get_environment_memory,
    get_observation_memory,
)
from app.schemas import ExperienceSchema
from fastapi import APIRouter, Depends, HTTPException, Path

router = APIRouter(prefix="/experience", tags=["experience"])


def _get_experience(
    train_id: str,
    worker_id: int,
    episode_id: int,
    step: int,
    environment_memory: Memory,
    observation_memory: Memory,
    action_memory: Memory,
) -> ExperienceSchema | None:
    key = (train_id, worker_id, episode_id, step)
    environment = environment_memory.get(key)
    observation = observation_memory.get(key)
    action = action_memory.get(key)
    next_observation = observation_memory.get(
        (train_id, worker_id, episode_id, step + 1)
    )
    """Aggregate and retrieve a complete experience tuple from memory.

    Combines data from three independent memory channels (observation, action,
    environment) to construct a complete RL experience tuple. Validates temporal
    consistency and component availability before returning.

    Args:
        train_id: Training session identifier.
        worker_id: Running instance ID for parallel workers (0-indexed).
        episode_id: Episode number within the training session.
        step: Time step within the episode (0-indexed).
        environment_memory: Memory instance for environment state messages.
        observation_memory: Memory instance for observation messages.
        action_memory: Memory instance for action messages.

    Returns:
        ExperienceSchema | None: Complete experience tuple if all components
            are available, None otherwise. Returns None if:
            - Environment state is missing (not published or evicted)
            - Observation at step t is missing
            - Action at step t is missing
            - Next observation at step t+1 is missing (for non-terminal states)
    """
    if (None in (environment, observation, action)) or (
        not environment.done and next_observation is None
    ):
        return None
    else:
        return ExperienceSchema(
            observation=observation,
            action=action,
            environment=environment,
            next_observation=next_observation,
        )


@router.get(
    "/{train_id}/{worker_id}/{episode_id}/{step}",
    summary="Retrieve complete experience tuple for RL training",
    description=(
        "Aggregates observation, action, and environment data into a complete "
        "experience tuple (s, a, r, done, s') for reinforcement learning training. "
    ),
    response_description="Complete experience tuple with state transition information",
    response_model=ExperienceSchema,
    responses={
        200: {
            "description": "Experience retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "observation": {
                            "data": [0.1, 0.5, 0.2, -0.3],
                        },
                        "action": {
                            "action": 1,
                        },
                        "environment": {"reward": 1.0, "done": False, "data": {}},
                        "next_observation": {
                            "data": [0.12, 0.48, 0.25, -0.28],
                        },
                    }
                }
            },
        },
        404: {
            "description": "Experience incomplete (missing components)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Experience not found for train_id=exp_001, worker_id=0, episode_id=5, step=10"
                    }
                }
            },
        },
    },
    tags=["experience"],
    operation_id="get_experience_step",
)
def get_experience_step(
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
    environment_memory: Memory = Depends(get_environment_memory),
    observation_memory: Memory = Depends(get_observation_memory),
    action_memory: Memory = Depends(get_action_memory),
) -> ExperienceSchema:
    """Retrieve a complete experience tuple for reinforcement learning training.

    This endpoint aggregates messages from three separate communication channels
    (observation, action, environment) to construct a complete experience tuple
    containing the standard state transition format.

    Args:
        train_id: Unique identifier for the training session/experiment.
        worker_id: Running instance number (for parallel environment execution).
        episode_id: Episode number within the training session.
        step: Time step to retrieve experience for.
        environment_memory: Injected environment memory instance (dependency injection).
        observation_memory: Injected observation memory instance (dependency injection).
        action_memory: Injected action memory instance (dependency injection).

    Returns:
        ExperienceSchema: Complete experience tuple containing:
            - observation: State at time t
            - action: Action taken at time t
            - environment: Reward, done flag, and other data at time t
            - next_observation: State at time t+1 (None if terminal state)

    Raises:
        HTTPException: 404 if experience is incomplete. Possible reasons:
            - Observation at step t not published or evicted from memory
            - Action at step t not published or evicted from memory
            - Environment state at step t not published or evicted
            - Next observation at step t+1 missing for non-terminal state
            - Components published in wrong order (timing issue)
    """
    experience = _get_experience(
        train_id,
        worker_id,
        episode_id,
        step,
        environment_memory,
        observation_memory,
        action_memory,
    )

    if experience is None:
        raise HTTPException(
            status_code=404,
            detail=f"Experience not found for train_id={train_id}, worker_id={worker_id}, episode_id={episode_id}, step={step}",
        )

    return experience
