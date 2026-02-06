"""
Pydantic schemas for reinforcement learning environment interactions.

This module defines the data models used for communication between RL components.
It includes schemas for observations, actions and environment states.
"""

from typing import Any

from pydantic import BaseModel


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
        executed: Boolean flag indicating if the action has been executed.
    """

    action: int
    executed: bool = False


class SuccessResponseSchema(BaseModel):
    """
    Represents a generic success response for API endpoints.

    Attributes:
        status: A string indicating the success status of the operation.
    """

    status: str = "success"
