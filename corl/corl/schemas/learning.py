from typing import Any

from pydantic import BaseModel, Field


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


class Action(BaseModel):
    """
    Represents an action to be taken in the environment.

    Attributes:
        action: List of float values encoding the action (supports both
            discrete actions encoded as ``[float(index)]`` and continuous
            action vectors).
    """

    action: list[float]


class Observation(BaseModel):
    """
    Represents an observation from the environment.

    Attributes:
        data: Dictionary containing observation data (e.g., sensor readings, state)
    """

    data: dict[str, Any] = Field(default_factory=dict)
