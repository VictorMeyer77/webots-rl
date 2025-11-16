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

import numpy as np
from brain.environment import Environment
from brain.model.monte_carlo import ModelMonteCarlo
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
        rewards: Mapping from (state_index, action) to list of sampled returns.
    """

    action_size: int
    observation_size: int
    observation_cardinality: int
    gamma: float
    epochs: int
    epsilon: float
    rewards: dict[tuple[int, int], list[float]] = {}
    model: ModelMonteCarlo | None

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
        self.gamma = gamma
        self.epsilon = epsilon
        self.rewards = {}
        self.model = ModelMonteCarlo(observation_cardinality=observation_cardinality)
        self.model.q_table = np.zeros((observation_cardinality**observation_size, action_size))

    def update_q_table(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> float:
        """
        Perform every-visit Monte Carlo update over one episode trajectory.

        Parameters:
            observations: ndarray of observation vectors.
            actions: ndarray of actions taken at each observation.
            rewards: ndarray of scalar rewards aligned with observations/actions.

        Returns:
            The return (discounted sum of rewards) from the first time step.
        """
        logger().debug(
            f"Monte Carlo update with {observations.shape[0]} observations, "
            f"{actions.shape[0]} actions, {rewards.shape[0]} rewards"
        )
        g = 0.0
        visited = set()
        for i in reversed(range(actions.shape[0])):
            g = rewards[i] + self.gamma * g
            observation_index = self.model.observation_to_index(observations[i])
            q_key = (observation_index, actions[i])
            if q_key not in visited:
                visited.add(q_key)
                if q_key not in self.rewards:
                    self.rewards[q_key] = []
                self.rewards[q_key].append(g)
                self.model.q_table[observation_index][actions[i]] = sum(self.rewards[q_key]) / len(self.rewards[q_key])
        return g

    def run(self, epochs: int) -> None:
        """
        Execute multiple training epochs.

        Per epoch:
            1. Decay epsilon.
            2. Generate an episode via simulation().
            3. Update Q-table with Monte Carlo returns.
            4. Log metrics and reset environment.

        Parameters:
            epochs: Number of training iterations.
        """
        for epoch in range(epochs):
            self.epsilon = max(0.01, self.epsilon * 0.998)  # todo set as parameters
            observations, actions, rewards = self.simulation()
            g = self.update_q_table(observations, actions, rewards)
            self.tb_writer.add_scalar("MonteCarlo/Return", g, epoch)
            self.environment.reset()
            logger().info(
                f"Epoch {epoch + 1}/{epochs} completed with return {g:.4f}, "
                f"epsilon {self.epsilon:.4f} and reward {sum(rewards)}"
            )
        self.close_tb()

    def save_model(self) -> None:
        """
        Persist the Q-table to disk using NumPy binary format (.npy).

        Side Effects:
            Writes file to self.model_path and logs success.
        """
        self.model.save(self.model_name)

    @abstractmethod
    def simulation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate one full episode trajectory.

        Returns:
            Tuple of (observations, actions, rewards):
                observations: ndarray of observation vectors.
                actions: ndarray of action indices.
                rewards: ndarray of float rewards.
        Raises:
            NotImplementedError: Must be implemented by subclasses integrating the environment.
        """
        raise NotImplementedError("Method simulation() not implemented in TrainerMonteCarlo.")
