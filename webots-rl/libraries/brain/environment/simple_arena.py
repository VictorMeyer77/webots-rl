from brain.environment import Environment, EnvironmentState
from brain.utils.logger import logger
from controller import Supervisor

EPUCK_DEF = "EPUCK"
FINISH_LINE_DEF = "FINISH_LINE"


class StateSimpleArena(EnvironmentState):
    """
    Represents the state of the simple arena environment.

    Extends:
        EnvironmentState

    Attributes:
        finish_line_distance (float): The distance from the agent to the finish line.
    """

    finish_line_distance: float

    def __init__(self):
        """
        Initializes the state with default values.
        Sets finish_line_distance to 99.
        """
        super().__init__()
        self.finish_line_distance = 99


class EnvironmentSimpleArena(Environment):
    """
    Environment for a simple arena simulation.

    Manages the agent and finish line positions, handles environment reset,
    runs the simulation loop, and computes the agent's state and distance to the finish line.

    Args:
        supervisor (Supervisor): Webots Supervisor instance.
        max_step (int): Maximum number of steps per episode.
    """

    def __init__(self, supervisor: Supervisor, max_step: int):
        """
        Initializes the SimpleArena environment.

        Retrieves references to the agent and finish line objects in the simulation.
        """
        super().__init__(supervisor, max_step)
        epuck = self.supervisor.getFromDef(EPUCK_DEF)
        self.epuck_translation = epuck.getField("translation")
        finish_line = self.supervisor.getFromDef(FINISH_LINE_DEF)
        self.finish_line_translation = finish_line.getField("translation")

    def run(self) -> float:
        """
        Runs the environment simulation loop.

        Returns:
            float: The total accumulated reward for the episode.
        """
        timestep = int(self.supervisor.getBasicTimeStep())
        total_reward = 0.0
        state = None

        while self.supervisor.step(timestep) != -1:
            state, step_reward = self.step()
            total_reward += step_reward
            if state.is_terminated:
                break
            logger().debug(f"Step {self.step_index}: Distance to finish line: {state.finish_line_distance:.4f}")
            self.step_index += 1

        logger().info(f"Simulation terminated at step {state.step_index}, success: {state.is_success}")
        return total_reward / self.max_step

    def finsh_distance(self) -> float:
        """
        Computes the Euclidean distance between the agent and the finish line.

        Returns:
            float: The distance to the finish line.
        """
        epuck_position = self.epuck_translation.getSFVec3f()
        finish_line_position = self.finish_line_translation.getSFVec3f()
        dx = finish_line_position[0] - epuck_position[0]
        dy = finish_line_position[1] - epuck_position[1]
        distance = (dx**2 + dy**2) ** 0.5
        return distance

    def state(self) -> StateSimpleArena:
        """
        Returns the current state of the environment.

        Returns:
            StateSimpleArena: The current environment state.
        """
        state = StateSimpleArena()
        state.finish_line_distance = self.finsh_distance()
        state.step_index = self.step_index
        state.is_terminated = state.finish_line_distance < 0.02 or self.step_index >= self.max_step
        state.is_success = state.finish_line_distance < 0.02
        return state

    def step(self) -> tuple[StateSimpleArena, float]:
        """
        Perform one step in the environment.

        On the first step, sends the agent's actions over TCP.
        Calculates the reward as the negative distance to the finish line,
        adds a large bonus if the task is successful, and applies a small
        penalty for each step taken.

        Returns:
            tuple[StateSimpleArena, float]: The new environment state and the computed reward.
        """

        state = self.state()
        reward = -state.finish_line_distance
        if state.is_success:
            reward += 10.0
        reward -= 0.0005 * state.step_index

        return state, reward
