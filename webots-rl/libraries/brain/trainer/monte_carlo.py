"""
Monte Carlo trainer implementing an every-visit epsilon-greedy control algorithm.

Core concepts:
- States are encoded as mixed-radix indices based on discrete, binned sensor values.
- Q-table shape: ``(observation_cardinality ** observation_size, action_size)``.
- Every-visit returns: for each (state, action) pair in an episode, the full discounted
  return is appended and the Q-value is updated as the mean of all stored returns.
- Epsilon decays multiplicatively per epoch until a floor value (0.01).

This trainer is built on top of :class:`ModelQTable`, which provides a generic
NumPy-backed Q-table and observation-to-index mapping usable by multiple
algorithms (Monte Carlo, SARSA, Q-learning, etc.).
"""

from abc import abstractmethod

import numpy as np
import tensorflow as tf
from brain.environment import Environment
from brain.model.q_table import ModelQTable
from brain.trainer import Trainer
from brain.utils.logger import logger


class TrainerMonteCarlo(Trainer):
    """Monte Carlo control trainer with an epsilon-greedy tabular policy.

    Attributes:
        action_size (int): Number of discrete actions.
        observation_size (int): Number of sensor readings composing a state.
        observation_cardinality (int): Number of discrete values each sensor can take.
        gamma (float): Discount factor (0 <= gamma <= 1).
        epsilon (float): Exploration rate for epsilon-greedy policy.
        epsilon_decay (float): Multiplicative decay factor for epsilon per epoch.
        rewards (dict[tuple[int, int], list[float]]): Mapping from
            ``(state_index, action)`` to a list of sampled returns ``G_t``.
        model (ModelQTable | None): Backing Q-table model shared with the controller.
    """

    action_size: int
    observation_size: int
    observation_cardinality: int
    gamma: float
    epochs: int
    epsilon: float
    epsilon_decay: float
    rewards: dict[tuple[int, int], list[float]] = {}
    model: ModelQTable | None

    def __init__(
        self,
        environment: Environment,
        model_name: str,
        action_size: int,
        observation_size: int,
        observation_cardinality: int,
        gamma: float,
        epsilon: float,
        epsilon_decay: float,
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
            epsilon_decay: Multiplicative decay factor for epsilon per epoch.
        """
        super().__init__(environment=environment, model_name=model_name)
        self.action_size = action_size
        self.observation_size = observation_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rewards = {}
        self.model = ModelQTable(observation_cardinality=observation_cardinality)
        self.model.q_table = np.zeros((observation_cardinality**observation_size, action_size))

    def update_q_table(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> float:
        """Perform every-visit Monte Carlo update over one episode.

        The episode is provided as three aligned arrays of equal length:
        ``observations[t]``, ``actions[t]``, and ``rewards[t]`` correspond to
        the state, action, and reward at time step ``t``. For each distinct
        ``(state_index, action)`` pair that appears in the episode, the method
        computes the full discounted return ``G_t`` from that time step onward
        and updates the corresponding Q-value as the mean of all sampled
        returns for that pair.

        Args:
            observations (np.ndarray): Array of observation vectors for each
                time step in the episode.
            actions (np.ndarray): Array of integer action indices, one per
                time step.
            rewards (np.ndarray): Array of scalar rewards aligned with
                ``observations`` and ``actions``.

        Returns:
            float: The return (discounted sum of rewards) from the first
            time step of the episode (``G_0``).
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
        """Execute multiple Monte Carlo training epochs.

        For each epoch:
          1. Decay ``epsilon`` (down to a minimum of 0.01).
          2. Generate one episode via :meth:`simulation`.
          3. Update the Q-table using :meth:`update_q_table`.
          4. Log the episode return and reset the environment.

        Args:
            epochs (int): Number of Monte Carlo training iterations (episodes).
        """
        for epoch in range(epochs):
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
            observations, actions, rewards = self.simulation()
            g = self.update_q_table(observations, actions, rewards)
            reward = sum(rewards)
            with self.tb_writer.as_default():
                tf.summary.scalar("MonteCarlo/Return", g, epoch)
                tf.summary.scalar("MonteCarlo/Reward", reward, epoch)
                tf.summary.scalar("MonteCarlo/Epsilon", self.epsilon, epoch)
            self.environment.reset()
            logger().info(
                f"Epoch {epoch + 1}/{epochs} completed with return {g:.4f}, "
                f"epsilon {self.epsilon:.4f} and reward {reward}"
            )
        self.close_tb()

    def save_model(self) -> None:
        """Persist the Q-table to disk using NumPy binary format (.npy).

        Side Effects:
            Writes a ``<model_name>.npy`` file under ``MODEL_PATH`` and logs success.
        """
        self.model.save(self.model_name)

    @abstractmethod
    def simulation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate one full episode trajectory.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                * observations: ndarray of observation vectors (one per step).
                * actions: ndarray of action indices.
                * rewards: ndarray of float rewards.

        Raises:
            NotImplementedError: Must be implemented by subclasses integrating
                a concrete environment and messaging protocol.
        """
        raise NotImplementedError("Method simulation() not implemented in TrainerMonteCarlo.")
