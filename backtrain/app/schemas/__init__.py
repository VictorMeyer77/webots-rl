"""
Pydantic schemas for reinforcement learning environment interactions.

This module defines the data models used for communication between RL components.
It includes schemas for observations, actions and environment states.
"""

from typing import Any

from pydantic import BaseModel, Field


class StepKeySchema(BaseModel):
    """
    Represents the unique key for a specific step in the training process.

    Attributes:
        train_id: Unique identifier for the training session/experiment.
        worker_id: Parallel worker instance identifier (0-indexed).
        episode_id: Episode number within the training session.
        step: Time step number within the episode.
    """

    train_id: str = Field(..., min_length=1)
    worker_id: int = Field(..., ge=0)
    episode_id: int = Field(..., ge=0)
    step: int = Field(..., ge=0)


class EnvironmentSchema(BaseModel):
    """
    Represents the state of the environment after an action is taken.

    Attributes:
        done: Boolean indicating whether the episode has terminated.
        reward: Numerical reward received from the environment.
        data: Additional environment-specific data as key-value pairs.
    """

    done: bool
    reward: float
    data: dict[str, Any]


class ObservationSchema(BaseModel):
    """
    Represents an observation from the environment made by the agent.

    Attributes:
        data: The observation data as key-value pairs, structure depends on the environment.
    """

    data: dict[str, Any]


class ActionSchema(BaseModel):
    """
    Represents an action to be taken in the environment.

    Attributes:
        action: Integer identifier for the action to execute.
        executed: Boolean flag indicating if the action has been executed (defaults to False).
    """

    action: list[float]


class SuccessResponseSchema(BaseModel):
    """
    Represents a generic success response for API endpoints.

    Attributes:
        status: A string indicating the success status of the operation (defaults to "success").
    """

    status: str = "success"
