import json

from brain.environment.model.genetic import EnvironmentGenetic
from brain.environment.simple_arena import SimpleArena, SimpleArenaState


class EnvironmentSimpleArenaGenetic(SimpleArena, EnvironmentGenetic):
    """
    Environment for a simple arena using a genetic algorithm.

    Inherits from SimpleArena and EnvironmentGenetic to provide
    step-based simulation logic for reinforcement learning or optimization.
    Sends initial actions over TCP at the first step, computes the reward
    based on the agent's distance to the finish line, success status, and
    penalizes longer episodes.

    Methods:
        step(): Advances the environment by one step and returns the new state and reward.

    """

    def step(self) -> tuple[SimpleArenaState, float]:
        """
        Perform one step in the environment.

        On the first step, sends the agent's actions over TCP.
        Calculates the reward as the negative distance to the finish line,
        adds a large bonus if the task is successful, and applies a small
        penalty for each step taken.

        Returns:
            tuple[SimpleArenaState, float]: The new environment state and the computed reward.
        """
        if self.step_index == 0:
            self.tcp_socket.send(json.dumps({"actions": self.actions}))

        state = self.state()
        reward = -state.finish_line_distance
        if state.is_success:
            reward += 10.0
        reward -= 0.0005 * state.step_index

        return state, reward
