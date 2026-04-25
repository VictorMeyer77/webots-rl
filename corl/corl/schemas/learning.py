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
    Represents an action to be executed in the environment.

    Attributes:
        action: The action identifier/value
        executed: Whether the action has been executed
    """

    action: int
    executed: bool = False


class Observation(BaseModel):
    """
    Represents an observation from the environment.

    Attributes:
        data: Dictionary containing observation data (e.g., sensor readings, state)
    """

    data: dict[str, Any] = Field(default_factory=dict)
