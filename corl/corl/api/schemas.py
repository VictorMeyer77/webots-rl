"""
Data schemas for reinforcement learning API communication.

This module defines Pydantic models for structured data exchange between
the RL environment, agents, and training backend.
"""

from typing import Any
from pydantic import BaseModel, Field
from enum import Enum


class Endpoint(str, Enum):
    """
    API endpoint identifiers for the RL training system.

    This enum defines the available REST API endpoints for communication
    between workers, environments, and the central training server.

    Attributes:
        ACTION: Endpoint for storing/retrieving agent actions
        OBSERVATION: Endpoint for storing/retrieving environment observations
        ENVIRONMENT: Endpoint for storing/retrieving environment state (reward, done)
        SUPERVISOR: Endpoint for training coordination (workers, episodes, etc.)
    """

    ACTION = "action"
    OBSERVATION = "observation"
    ENVIRONMENT = "environment"
    SUPERVISOR = "supervisor"


class Environment(BaseModel):
    """
    Represents the environment state in a reinforcement learning step.

    Attributes:
        done: Whether the episode has terminated
        reward: The reward received for the current step
        data: Additional environment-specific metadata
    """

    done: bool = False
    reward: float = 0.0
    data: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization (backwards compatible)."""
        return self.model_dump()


class Action(BaseModel):
    """
    Represents an action to be executed in the environment.

    Attributes:
        action: The action identifier/value
        executed: Whether the action has been executed
    """

    action: int
    executed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization (backwards compatible)."""
        return self.model_dump()


class Observation(BaseModel):
    """
    Represents an observation from the environment.

    Attributes:
        data: Dictionary containing observation data (e.g., sensor readings, state)
    """

    data: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization (backwards compatible)."""
        return self.model_dump()
