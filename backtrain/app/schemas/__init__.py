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
        executed: Boolean flag indicating if the action has been executed.
    """

    action: int
    executed: bool = False


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


class TrainIdSchema(BaseModel):
    """Request model for creating a new training session."""

    train_id: str


class WorkerIdSchema(BaseModel):
    """Response model for adding a worker to a training session."""

    worker_id: int


class EpisodeIdSchema(BaseModel):
    """Response model for retrieving an episode ID."""

    episode_id: int


class SystemInfoSchema(BaseModel):
    """Response model for system information endpoint."""

    supervisor: dict[str, dict[int, int]]
    action_memory: dict[str, int | float]
    environment_memory: dict[str, int | float]
    observation_memory: dict[str, int | float]
