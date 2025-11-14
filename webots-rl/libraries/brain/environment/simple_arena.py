"""
Simple arena environment for Webots reinforcement learning.

This module defines:
- `StateSimpleArena`: State dataclass for the agent in the arena, including distance to the finish line.
- `EnvironmentSimpleArena`: Environment class managing agent and finish line positions, simulation loop,
  state computation, reward calculation, and environment reset.

The environment expects Webots Supervisor API access and is designed for discrete-step RL training.
"""

from dataclasses import dataclass

from brain.environment import Environment, EnvironmentState
from brain.utils.logger import logger
from controller import Supervisor

EPUCK_DEF = "EPUCK"
FINISH_LINE_DEF = "FINISH_LINE"


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
    Webots environment for a simple arena task.

    Handles agent and finish line references, computes state and reward, and manages
    the simulation loop for RL training.

    Args:
        supervisor (Supervisor): Webots Supervisor instance.
        timestep (int): Simulation timestep in milliseconds.
        max_step (int): Maximum number of steps per episode.
    """

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
        state.is_terminated = state.finish_line_distance < 0.025 or self.step_index >= self.max_step - 1
        state.is_success = state.finish_line_distance < 0.025
        return state

    def step(self) -> tuple[StateSimpleArena, float]:
        """
        Perform one environment step: compute state and reward.

        Reward is negative distance to finish line, with a bonus for success and a small step penalty.

        Returns:
            tuple[StateSimpleArena, float]: (New state, computed reward)
        """
        state = self.state()
        reward = -state.finish_line_distance
        if state.is_success:
            reward += 10.0
        reward -= 0.0005 * state.step_index
        return state, reward

    def reset(self):
        """
        Reset the environment and restart the agent controller.
        """
        self.epuck.restartController()
        super().reset()
