"""
Pydantic schemas for reinforcement learning environment interactions.

This module defines the data models used for communication between RL components.
It includes schemas for observations, actions, environment
states, and complete experience tuples used in training.
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
    """

    action: int


class ExperienceSchema(BaseModel):
    """
    Represents a single experience tuple from an RL interaction.

    This schema captures a complete transition in the environment, which is
    the fundamental unit for reinforcement learning algorithms.

    Attributes:
        observation: The state observed before taking the action.
        action: The action taken by the agent.
        environment: The resulting environment state, including reward and done flag.
        next_observation: The state observed after taking the action, or None if episode ended.
    """

    observation: ObservationSchema
    action: ActionSchema
    environment: EnvironmentSchema
    next_observation: ObservationSchema | None
