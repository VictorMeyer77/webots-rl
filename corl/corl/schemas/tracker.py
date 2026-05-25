import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


class StepKey(BaseModel):
    """
    Unique identifier for a specific step in the training process.

    This class provides a frozen (immutable) identifier that can be hashed and
    compared, making it suitable for use as dictionary keys or in sets. Instances
    are ordered first by worker_id, then episode_id, then step.

    Attributes:
        worker_id: Unique identifier for the worker process (must be non-negative).
        episode_id: Current episode number for this worker (must be non-negative).
        step: Step number within the current episode (must be non-negative).
    """

    model_config = ConfigDict(frozen=True)

    worker_id: int = Field(..., ge=0, description="Unique identifier for the worker")
    episode_id: int = Field(..., ge=0, description="Episode number for this worker")
    step: int = Field(..., ge=0, description="Step number within the episode")

    def __hash__(self) -> int:
        return hash((self.worker_id, self.episode_id, self.step))

    def __lt__(self, other: "StepKey") -> bool:
        return (self.worker_id, self.episode_id, self.step) < (
            other.worker_id,
            other.episode_id,
            other.step,
        )


class StepResult(BaseModel):
    """
    Result data from a single environment step.

    Contains the observation, action taken, reward received, and terminal flag
    for a single timestep in an episode. All fields are optional to support
    incremental construction of results during async step execution.

    Attributes:
        observation: Environment observation (sensor data, state) as a numpy array.
            Can be None if not yet received.
        action: List of floats representing the action taken by the agent.
            Single-element for discrete actions (e.g. ``[3.0]``), multi-element
            for continuous actions (e.g. ``[0.5, -0.3]``).
            Can be None if not yet determined.
        reward: Scalar reward received from the environment.
            Can be None if not yet computed.
        done: Whether this step terminates the episode.
            Can be None if not yet determined.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    observation: NDArray[np.float32] | None = Field(
        default=None, description="Environment observation array"
    )
    action: list[float] | None = Field(
        default=None, description="Action taken by agent"
    )
    reward: float | None = Field(
        default=None, description="Reward received from environment"
    )
    done: bool | None = Field(default=None, description="Whether episode is terminated")

    def is_complete(self) -> bool:
        """
        Check if all required fields have been populated.

        Returns:
            bool: True if observation, action, reward, and done are all non-None,
                False otherwise.
        """
        return all(
            [
                self.observation is not None,
                self.action is not None,
                self.reward is not None,
                self.done is not None,
            ]
        )


class Transition(BaseModel):
    """
    A temporal-difference (TD) transition pair used for training.

    Bundles a consecutive pair of :class:`StepResult` objects — the step at
    time *t* and the step at time *t+1* — into a single unit suitable for
    computing TD targets (e.g. Q-learning, actor-critic updates).

    ``next_step`` is ``None`` for terminal transitions, i.e. when
    ``current_step.done`` is ``True`` and there is no following state.

    Attributes:
        current_step: The step result at time *t*, including observation,
            action, reward, and done flag.
        next_step: The step result at time *t+1*, or ``None`` if
            ``current_step`` is the last step of an episode.
    """

    current_step: StepResult
    next_step: StepResult | None
