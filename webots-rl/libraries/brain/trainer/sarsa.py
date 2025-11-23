"""SARSA training module.

Implements an on-policy Temporal Difference (TD) control algorithm (SARSA)
for discrete observation and action spaces using a tabular Q-table.

Key Concepts:
  * SARSA update: Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') - Q(s,a) ]
    where the next action a' is selected by the current (ε-greedy) policy.
  * Exploration: ε decays each epoch down to a minimum (0.01) to balance
    exploration vs exploitation.
  * Representation: Observations are mapped to integer indices in the
    Q-table via ModelQTable.observation_to_index() assuming each component
    has identical cardinality.

Usage:
  Subclass TrainerSarsa and implement simulation(), performing one full
  episode loop:
      1. Initialize observation and choose action (ε-greedy).
      2. Step environment -> (next_observation, reward, terminated).
      3. Choose next_action (ε-greedy) if not terminated.
      4. Call update_q_table(...).
      5. Accumulate total reward.
      6. Return total reward (float) to be logged.
"""

from abc import abstractmethod

import numpy as np
from brain.environment import Environment
from brain.model.q_table import ModelQTable
from brain.trainer import Trainer
from brain.utils.logger import logger


class TrainerSarsa(Trainer):
    """
    SARSA trainer handling Q-table initialization, per-epoch execution,
    logging, and model persistence.

    Attributes:
        action_size (int): Number of discrete actions.
        observation_size (int): Length of the observation vector.
        observation_cardinality (int): Number of discrete values per observation component.
        alpha (float): Learning rate (step size) for TD updates.
        gamma (float): Discount factor for future returns.
        epsilon (float): Current ε for ε-greedy policy (decays each epoch).
        model (ModelQTable | None): Wrapper holding the Q-table and indexing utilities.
    """

    action_size: int
    observation_size: int
    observation_cardinality: int
    alpha: float
    gamma: float
    epochs: int
    epsilon: float
    model: ModelQTable | None

    def __init__(
        self,
        environment: Environment,
        model_name: str,
        action_size: int,
        observation_size: int,
        observation_cardinality: int,
        alpha: float,
        gamma: float,
        epsilon: float,
    ):
        """
        Initialize trainer and allocate Q-table.

        Args:
            environment (Environment): Interactive environment instance.
            model_name (str): Stem used when saving the model.
            action_size (int): Number of discrete actions.
            observation_size (int): Length of observation vector.
            observation_cardinality (int): Discrete values per observation feature.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Initial ε for ε-greedy policy.
        """
        super().__init__(environment=environment, model_name=model_name)
        self.action_size = action_size
        self.observation_size = observation_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = ModelQTable(observation_cardinality=observation_cardinality)
        self.model.q_table = np.zeros((observation_cardinality**observation_size, action_size))

    def update_q_table(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        next_action: int,
        terminated: bool = False,
    ) -> None:
        """Apply a single SARSA TD update to the Q-table.

        This method updates the Q-value associated with a given state–action
        pair ``(observation, action)`` using the SARSA rule:

            Q(s, a) ← Q(s, a) + α [ r + γ Q(s', a') − Q(s, a) ]

        where ``(s, a)`` is the current state–action pair, ``r`` is the
        immediate reward, and ``(s', a')`` is the next state–action pair
        chosen according to the current ε-greedy policy. When the transition
        leads to a terminal state, the future value term ``γ Q(s', a')`` is
        omitted and the target reduces to ``r``.

        Args:
            observation (np.ndarray): Current state observation before taking
                ``action``. This is mapped to a discrete index via
                ``ModelQTable.observation_to_index``.
            action (int): Index of the action taken in the current state.
            reward (float): Immediate scalar reward obtained after executing
                ``action`` in ``observation``.
            next_observation (np.ndarray): Observation of the next state
                after the transition. Only used when ``terminated`` is
                ``False``.
            next_action (int): Index of the next action selected in
                ``next_observation`` by the current ε-greedy policy. Only
                used when ``terminated`` is ``False``.
            terminated (bool, optional): Flag indicating whether the
                transition ended the episode. If ``True``, no bootstrap term
                is added and the TD target is equal to ``reward``.

        Returns:
            None: The Q-table stored in ``self.model.q_table`` is updated
            in-place.
        """

        obs_index = self.model.observation_to_index(observation)
        if terminated:
            td_target = reward
        else:
            next_obs_index = self.model.observation_to_index(next_observation)
            td_target = reward + self.gamma * self.model.q_table[next_obs_index][next_action]
        td_error = td_target - self.model.q_table[obs_index][action]
        self.model.q_table[obs_index][action] += self.alpha * td_error

    def run(self, epochs: int) -> None:
        """
        Execute multiple training epochs.

        Process per epoch:
          1. Decay ε.
          2. Run one episode via simulation().
          3. Log reward and ε to TensorBoard.
          4. Reset environment.

        Args:
            epochs (int): Number of training episodes.
        """
        for epoch in range(epochs):
            self.epsilon = max(0.01, self.epsilon * 0.999)
            reward = self.simulation()
            self.tb_writer.add_scalar("Sarsa/Reward", reward, epoch)
            self.tb_writer.add_scalar("Sarsa/Epsilon", self.epsilon, epoch)
            self.environment.reset()
            logger().info(
                f"Epoch {epoch + 1}/{epochs} completed with " f"epsilon {self.epsilon:.4f} and reward {reward}"
            )
        self.close_tb()

    def save_model(self) -> None:
        """
        Persist the Q-table to disk using NumPy binary format (.npy).
        """
        self.model.save(self.model_name)

    @abstractmethod
    def simulation(self) -> float:
        """
        Run a single episode trajectory (environment loop).

        Responsibilities for subclass implementation:
          * Initialize starting observation and select initial action (ε-greedy).
          * Iterate until terminal condition:
              - Perform environment step.
              - Select next action (if not terminal).
              - Call update_q_table().
              - Accumulate total reward.
          * Return total episode reward (float).

        Returns:
            float: Total accumulated reward for the episode.
        """
        raise NotImplementedError("Method simulation() not implemented in TrainerSarsa.")
