"""
Abstract environment and state base definitions for a Webots reinforcement learning framework.

This module provides:
- `EnvironmentState`: Lightweight dataclass capturing episode progression, termination, and success.
- `Environment`: Abstract base class defining the interface expected from any concrete Webots RL environment:
  `step()`, `state()`, `run()`, plus lifecycle helpers `reset()` and `quit()`.

Subclasses should:
- Implement `state()` to build and return a domain-specific `EnvironmentState` subtype (optional).
- Implement `step()` to apply one simulation tick and compute reward.
- Implement `run()` to execute an episode loop until termination.

Attributes expected in concrete environments:
- `supervisor`: Webots `Supervisor` controlling the simulation.
- `timestep`: Simulation step duration (ms).
- `step_index`: Current discrete step counter.
- `max_step`: Maximum allowed steps per episode.
- `queue`: Message queue utility for controller/supervisor communication (may be `None` if unused).
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from brain.utils.logger import logger
from brain.utils.queue import Queue
from controller import Supervisor


@dataclass
class EnvironmentState:
    """
    Serializable base state for an environment episode.

    Tracks termination and success flags plus the current step index. Can be
    subclassed to add domain-specific fields (e.g. distances, sensor arrays).

    Attributes:
        is_terminated (bool): True when episode must stop (goal reached or limit exceeded).
        is_success (bool): True when termination corresponds to goal achievement.
        step_index (int): Zero-based index of the current step within the episode.
    """

    is_terminated: bool = False
    is_success: bool = False
    step_index: int = 0

    def to_json(self) -> str:
        """
        Serialize the state to a JSON string.

        Returns:
            str: JSON representation of all instance attributes.
        """
        return json.dumps(self.__dict__)


class Environment(ABC):
    """
    Abstract base class for Webots reinforcement learning environments.

    Manages common simulation bookkeeping and enforces the interface for
    stepping, state retrieval, episode execution, and lifecycle management.

    Attributes:
        supervisor (Supervisor): Webots Supervisor instance used to control simulation.
        timestep (int): Simulation step duration in milliseconds.
        step_index (int): Current step number (auto-incremented externally or in `run()`).
        max_step (int): Maximum number of steps permitted per episode.
        queue (Queue | None): Communication helper for message-based interaction (optional).
    """

    supervisor: Supervisor
    timestep: int
    step_index: int
    max_step: int
    queue: Queue | None

    def __init__(self, supervisor: Supervisor, timestep: int, max_step: int):
        """
        Initialize the environment core structures.

        Args:
            supervisor (Supervisor): Active Webots Supervisor.
            timestep (int): Simulation timestep in milliseconds.
            max_step (int): Upper bound on steps per episode.
        """
        self.supervisor = supervisor
        self.timestep = timestep
        self.step_index = 0
        self.max_step = max_step
        self.queue = Queue(timestep, supervisor.getDevice("emitter"), supervisor.getDevice("receiver"))
        logger().debug(f"Environment initialized, max steps: {self.max_step}")

    @abstractmethod
    def step(self) -> tuple[EnvironmentState, float]:
        """
        Advance the simulation by one logical RL step.

        Must:
            - Update or compute the new `EnvironmentState`.
            - Compute and return the reward for the transition.

        Returns:
            tuple[EnvironmentState, float]: New state and scalar reward.
        """
        raise NotImplementedError("Method step() not implemented.")

    @abstractmethod
    def state(self) -> EnvironmentState:
        """
        Build and return the current state snapshot.

        Returns:
            EnvironmentState: Current environment state (may be subclass instance).
        """
        raise NotImplementedError("Method state() not implemented.")

    def reset(self) -> None:
        """
        Reset simulation and internal counters to episode start.
        """
        self.supervisor.simulationReset()
        self.step_index = 0
        logger().info("Environment reset.")

    def quit(self) -> None:
        """
        Terminate the simulation process.
        """
        self.supervisor.simulationQuit(0)
        logger().info("Environment terminated successfully.")

    @abstractmethod
    def run(self) -> EnvironmentState:
        """
        Execute an episode loop until termination.

        Expected pattern:
            - While not terminated: call `step()`, increment `step_index`.
            - Return the final `EnvironmentState`.

        Returns:
            EnvironmentState: Terminal state at episode completion.
        """
        raise NotImplementedError("Method run() not implemented.")

    @abstractmethod
    def randomize(self) -> None:
        """Apply environment-specific randomization before or between episodes.

        Typical uses include:
          * Randomizing the agent's start pose (position/orientation).
          * Randomizing target or obstacle positions.
          * Sampling domain parameters (e.g. noise levels, textures).

        This method is intended to be called by trainers before ``reset()``
        or at the beginning of an episode to increase diversity and
        robustness of learned policies.
        """
        raise NotImplementedError("Method randomize() not implemented.")
