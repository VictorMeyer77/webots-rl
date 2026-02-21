import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class StepKey(BaseModel):
    """
    Unique identifier for a specific step in the training process.

    Attributes:
        worker_id: Unique identifier for the worker process (must be non-negative)
        episode_id: Current episode number for this worker (must be non-negative)
        step: Step number within the current episode (must be non-negative)
    """

    model_config = ConfigDict(frozen=True)

    worker_id: int = Field(..., ge=0, description="Unique identifier for the worker")
    episode_id: int = Field(..., ge=0, description="Episode number for this worker")
    step: int = Field(..., ge=0, description="Step number within the episode")


class StepResult(BaseModel):
    """
    Result data from a single environment step.

    Contains the observation, action taken, reward received, and terminal flag
    for a single timestep in an episode.

    Attributes:
        observation: Environment observation (sensor data, state)
        action: Action taken by the agent (discrete index or continuous values)
        reward: Scalar reward received from the environment
        done: Whether this step terminates the episode
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    observation: np.ndarray = Field(..., description="Environment observation array")
    action: int = Field(..., ge=0, description="Action index taken by agent")
    reward: float = Field(..., description="Reward received from environment")
    done: bool = Field(..., description="Whether episode is terminated")
