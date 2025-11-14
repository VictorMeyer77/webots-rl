"""
Monte Carlo trainer implementing an every-visit epsilon-greedy control algorithm.

Core concepts:
- States are encoded as mixed-radix indices based on discrete sensor values.
- Q-table shape: (observation_cardinality ** observation_size, action_size).
- Every-visit returns: for each (state, action) pair in an episode, the full discounted return
  is appended and the Q-value is updated as the mean of all stored returns.
- Epsilon decays multiplicatively per epoch until a floor value (0.01).
"""

from abc import abstractmethod
from typing import Any

import numpy as np
from brain.environment import Environment
from brain.train_model import Trainer
from brain.utils.logger import logger


class TrainerMonteCarlo(Trainer):
    """
    Monte Carlo control trainer with an epsilon-greedy policy over a tabular Q-function.

    Attributes:
        action_size: Number of discrete actions.
        observation_size: Number of sensor readings composing a state.
        observation_cardinality: Number of discrete values each sensor can take.
        gamma: Discount factor (0 <= gamma <= 1).
        epsilon: Exploration rate for epsilon-greedy policy.
        q_table: NumPy array storing Q-values for all state-action pairs.
        rewards: Mapping from (state_index, action) to list of sampled returns.
    """

    action_size: int
    observation_size: int
    observation_cardinality: int
    gamma: float
    epochs: int
    epsilon: float
    q_table: Any
    rewards: dict[tuple[int, int], list[float]] = {}

    def __init__(
        self,
        environment: Environment,
        model_name: str,
        action_size: int,
        observation_size: int,
        observation_cardinality: int,
        gamma: float,
        epsilon: float,
    ):
        """
        Initialize the Monte Carlo trainer.

        Parameters:
            environment: Simulation environment providing reset and interaction.
            model_name: Name used for saving the Q-table model.
            action_size: Number of discrete actions.
            observation_size: Length of the observation vector.
            observation_cardinality: Number of discrete values per observation component.
            gamma: Discount factor applied to future rewards.
            epsilon: Initial exploration rate for the epsilon-greedy policy.
        """
        super().__init__(environment=environment, model_name=model_name)
        self.action_size = action_size
        self.observation_size = observation_size
        self.observation_cardinality = observation_cardinality
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((observation_cardinality**observation_size, action_size))
        self.rewards = {}

    def index_to_observation(self, index: int) -> list[int]:
        """
        Decode a flattened state index back into its observation vector.

        Parameters:
            index: Integer index representing a state.

        Returns:
            A list of length observation_size with each component in [0, observation_cardinality - 1].
        """
        observation = []
        for _ in range(self.observation_size):
            observation.append(index % self.observation_cardinality)
            index //= self.observation_cardinality
        return observation

    def observation_to_index(self, observation: list[int]) -> int:
        """
        Encode an observation vector into a single integer index.

        Parameters:
            observation: List of discrete sensor values.

        Returns:
            Integer index usable to address the Q-table.
        """
        index = 0
        mul = 1
        for obs in observation:
            index += obs * mul
            mul *= self.observation_cardinality
        return index

    def update_q_table(self, observations: list[list[int]], actions: list[int], rewards: list[float]) -> float:
        """
        Perform every-visit Monte Carlo update over one episode trajectory.

        Parameters:
            observations: Sequence of observation vectors.
            actions: Sequence of actions taken at each observation.
            rewards: Sequence of scalar rewards aligned with observations/actions.

        Returns:
            The return (discounted sum of rewards) from the first time step.
        """
        logger().debug(
            f"Monte Carlo update with {len(observations)} observations, {len(actions)} actions, {len(rewards)} rewards"
        )
        g = 0.0
        visited = set()
        for i in reversed(range(len(actions))):
            g = rewards[i] + self.gamma * g
            observation_index = self.observation_to_index(observations[i])
            if (observation_index, actions[i]) not in visited:
                visited.add((observation_index, actions[i]))
                if (observation_index, actions[i]) not in self.rewards:
                    self.rewards[(observation_index, actions[i])] = []
                self.rewards[(observation_index, actions[i])].append(g)
                self.q_table[observation_index][actions[i]] = sum(self.rewards[(observation_index, actions[i])]) / len(
                    self.rewards[(observation_index, actions[i])]
                )
        return g

    def run(self, epochs: int):
        """
        Execute multiple training epochs.

        Per epoch:
            1. Decay epsilon.
            2. Generate an episode via simulation().
            3. Update Q-table with Monte Carlo returns.
            4. Log metrics and reset environment.

        Parameters:
            epochs: Number of training iterations.

        Returns:
            The learned Q-table (NumPy array).
        """
        for epoch in range(epochs):
            self.epsilon = max(0.01, self.epsilon * 0.998)
            observations, actions, rewards = self.simulation()
            g = self.update_q_table(observations, actions, rewards)
            self.tb_writer.add_scalar("MonteCarlo/Return", g, epoch)
            self.environment.reset()
            logger().info(f"Epoch {epoch + 1}/{epochs} completed with return {g:.4f}, epsilon {self.epsilon:.4f}")
        self.close_tb()

    def save_model(self):
        """
        Persist the Q-table to disk using NumPy binary format (.npy).

        Side Effects:
            Writes file to self.model_path and logs success.
        """
        np.save(self.model_path, self.q_table)
        logger().info(f"Model saved successfully at {self.model_path}")

    @abstractmethod
    def simulation(self) -> tuple[list[list[int]], list[int], list[float]]:
        """
        Generate one full episode trajectory.

        Returns:
            Tuple of (observations, actions, rewards):
                observations: list of observation vectors.
                actions: list of action indices.
                rewards: list of float rewards.
        Raises:
            NotImplementedError: Must be implemented by subclasses integrating the environment.
        """
        raise NotImplementedError("Method simulation() not implemented in TrainerMonteCarlo.")
