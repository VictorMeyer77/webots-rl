"""
Simple arena environment for Webots reinforcement learning.

Overview:
    A Supervisor-managed episodic task where an e-puck robot moves toward a finish line.
    The environment supplies:
      * `StateSimpleArena` dataclass: distance to target plus generic episode flags.
      * `EnvironmentSimpleArena`: distance computation, reward shaping, termination, reset, and
        a blocking `run()` loop for evaluation.

Entities (resolved by DEF names):
    * Robot: `EPUCK`
    * Finish line: `FINISH_LINE`

Termination criteria:
    Episode ends when either:
      * Distance to finish line < 0.025 (success).
      * Step index >= `max_step - 1` (timeout).
"""
import random
from dataclasses import dataclass
from typing import Any

from brain.environment import Environment, EnvironmentState
from brain.utils.logger import logger
from controller import Supervisor

EPUCK_DEF = "EPUCK"
FINISH_LINE_DEF = "FINISH_LINE"
FINISH_DISTANCE_THRESHOLD = 0.025


@dataclass
class StateSimpleArena(EnvironmentState):
    """
    State representation for the simple arena environment.

    Attributes:
        finish_line_distance (float): Euclidean distance from agent to finish line.
    """

    finish_line_distance: float = 99


class EnvironmentSimpleArena(Environment):
    """
    Target-approach Webots environment.

    Responsibilities:
        * Resolve robot and finish line supervisor nodes.
        * Compute distance and derive success / termination flags.
        * Produce shaped reward encouraging forward progress.
        * Provide blocking `run()` for evaluation and granular `step()` for trainers.

    Args:
        supervisor (Supervisor): Active Webots Supervisor.
        timestep (int): Simulation timestep in milliseconds.
        max_step (int): Episode step cap.
    """

    epuck: Any
    epuck_translation: Any
    finish_line_translation: Any
    last_distance: float | None
    initial_distance: float | None

    def __init__(self, supervisor: Supervisor, timestep: int, max_step: int):
        """
        Initialize the environment, retrieving agent and finish line nodes.

        Args:
            supervisor (Supervisor): Webots Supervisor instance.
            timestep (int): Simulation timestep.
            max_step (int): Maximum steps per episode.
        """
        super().__init__(supervisor, timestep, max_step)
        self.epuck = self.supervisor.getFromDef(EPUCK_DEF)
        self.epuck_translation = self.epuck.getField("translation")
        finish_line = self.supervisor.getFromDef(FINISH_LINE_DEF)
        self.finish_line_translation = finish_line.getField("translation")
        self.last_distance = None
        self.initial_distance = None

    def run(self) -> EnvironmentState:
        """
        Run the simulation loop until termination.

        Returns:
            EnvironmentState: Final state at episode end.
        """
        state = None
        while self.supervisor.step(self.timestep) != -1:
            state, step_reward = self.step()
            if state.is_terminated:
                break
            self.step_index += 1
        logger().info(f"Simulation terminated at step {state.step_index}, success: {state.is_success}")
        return state

    def finish_distance(self) -> float:
        """
        Compute the Euclidean distance between agent and finish line.

        Returns:
            float: Distance to the finish line.
        """
        epuck_position = self.epuck_translation.getSFVec3f()
        finish_line_position = self.finish_line_translation.getSFVec3f()
        dx = finish_line_position[0] - epuck_position[0]
        dy = finish_line_position[1] - epuck_position[1]
        distance = (dx**2 + dy**2) ** 0.5
        return distance

    def state(self) -> StateSimpleArena:
        """
        Get the current environment state.

        Returns:
            StateSimpleArena: Current state including distance and termination flags.
        """
        state = StateSimpleArena()
        state.finish_line_distance = self.finish_distance()
        state.step_index = self.step_index
        state.is_terminated = state.finish_line_distance < FINISH_DISTANCE_THRESHOLD or self.step_index >= self.max_step - 1
        state.is_success = state.finish_line_distance < FINISH_DISTANCE_THRESHOLD
        return state

    def step(self) -> tuple[StateSimpleArena, float]:
        """
        Advance environment one logical step (no direct action applied here).

        Reward:
            success -> +10.0
            timeout (no success) -> -2.0
            progress (>0) -> +0.5 * normalized_progress
            step penalty -> -0.0005

        Normalized progress:
            (last_distance - current_distance) / max(initial_distance, 1e-9)

        Returns:
            tuple[StateSimpleArena, float]: (state, reward)
        """
        state = self.state()
        reward = 0.0

        if state.is_success:
            reward += 10.0  # Success bonus
        elif state.is_terminated:
            reward -= 2.0  # Failure penalty
        else:
            if state.step_index > 0:
                progress = (self.last_distance - state.finish_line_distance) / max(self.initial_distance, 1e-9)
                if progress > 0:
                    reward += 0.5 * progress  # Progress reward
            else:
                self.initial_distance = state.finish_line_distance

        reward -= 0.0005  # Step penalty
        self.last_distance = state.finish_line_distance

        return state, reward

    def reset(self) -> None:
        """
        Reset episode bookkeeping and restart robot controller.

        Actions:
            * Restart e-puck controller.
            * Clear distance history.
            * Delegate base reset (clears step index and sets initial state).
        """
        self.epuck.restartController()
        self.last_distance = None
        self.initial_distance = None
        super().reset()

    def randomize(self) -> None:
        """Randomize the e-puck start position within the arena.

        The robot translation is uniformly sampled in the square
        ``x, y in [-0.45, 0.45]`` with a fixed ``z = 0.0``. If the
        sampled position is closer to the finish line than the
        ``FINISH_DISTANCE_THRESHOLD``, a new position is resampled
        recursively. This is intended to be called before a new
        training episode to add variety to starting conditions.
        """
        new_position = [random.uniform(-0.45, 0.45), random.uniform(-0.45, 0.45), 0.0]
        self.epuck_translation.setSFVec3f(new_position)
        if self.finish_distance() < FINISH_DISTANCE_THRESHOLD:
            self.randomize()
