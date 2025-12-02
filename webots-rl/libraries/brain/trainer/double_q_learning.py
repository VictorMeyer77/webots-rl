"""Double Q-learning training module.

Implements an off-policy Temporal Difference (TD) control algorithm
(Double Q-learning) for discrete observation and action spaces using two
separate tabular Q-tables.

Key Concepts:
  * Double Q-learning maintains two Q-tables, Q_A and Q_B, to reduce
    overestimation bias present in standard Q-learning.
  * On each update, one table is updated while the other is used to
    evaluate the greedy action selected from the first table.
  * Exploration: ε decays each epoch down to a minimum (0.01) to balance
    exploration vs exploitation.
  * Representation: Observations are mapped to integer indices in the
    Q-tables via ModelQTable.observation_to_index() assuming each component
    has identical cardinality.

Usage:
  Subclass TrainerDoubleQLearning and implement simulation(), performing one full
  episode loop:
      1. Observe the current state and select an action (ε-greedy) based on
         the two Q-tables (e.g. their sum or average).
      2. Step environment -> (next_observation, reward, terminated).
      3. Randomly choose which table ("A" or "B") to update and call
         update_q_table(...).
      4. Move to the next state and repeat until termination.
      5. Return total reward (float) to be logged.
"""

from abc import abstractmethod

import numpy as np
from brain.environment import Environment
from brain.model.q_table import ModelQTable
from brain.trainer import Trainer
from brain.utils.logger import logger


class TrainerDoubleQLearning(Trainer):
    """Double Q-learning trainer handling two Q-tables and persistence.

    This trainer maintains two tabular Q-functions (Q_A and Q_B) for a
    discrete observation and action space and updates them using the
    Double Q-learning rule. It also manages ε-greedy exploration,
    per-epoch execution, logging, and model saving.

    Attributes:
        action_size (int): Number of discrete actions.
        observation_size (int): Length of the observation vector.
        observation_cardinality (int): Number of discrete values per observation component.
        alpha (float): Learning rate (step size) for TD updates.
        gamma (float): Discount factor for future returns.
        epsilon (float): Current ε for ε-greedy policy (decays each epoch).
        epsilon_decay (float): Multiplicative decay factor for ε per episode.
        model_a (ModelQTable | None): Q-table model for Q_A.
        model_b (ModelQTable | None): Q-table model for Q_B.
    """

    action_size: int
    observation_size: int
    observation_cardinality: int
    alpha: float
    gamma: float
    epochs: int
    epsilon: float
    epsilon_decay: float
    model_a: ModelQTable | None
    model_b: ModelQTable | None

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
        epsilon_decay: float,
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
            epsilon_decay (float): Multiplicative decay factor for ε per episode.
        """
        super().__init__(environment=environment, model_name=model_name)
        self.action_size = action_size
        self.observation_size = observation_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.model_a = ModelQTable(observation_cardinality=observation_cardinality)
        self.model_a.q_table = np.zeros((observation_cardinality**observation_size, action_size))
        self.model_b = ModelQTable(observation_cardinality=observation_cardinality)
        self.model_b.q_table = np.zeros((observation_cardinality**observation_size, action_size))

    def update_q_table(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        table: str,
        terminated: bool = False,
    ) -> None:
        """Apply one Double Q-learning TD update to one of the Q-tables.

        Depending on ``table``, this method updates either Q_A or Q_B.
        When updating Q_A, the next greedy action is selected using Q_B
        to evaluate the action chosen by Q_A, and vice versa. This
        decouples action selection and evaluation, reducing overestimation
        bias relative to standard Q-learning.

        Concretely, for a non-terminal transition (s, a, r, s'):

            if table == "A":
                a* = argmax_a' Q_A(s', a')
                target = r + γ Q_B(s', a*)
            else:  # table == "B"
                a* = argmax_a' Q_B(s', a')
                target = r + γ Q_A(s', a*)

        For terminal next states, the target reduces to ``reward``.

        Args:
            observation (np.ndarray): Observation of the current state
                before taking ``action``. This is mapped to a discrete
                index via ``ModelQTable.observation_to_index``.
            action (int): Index of the action taken in the current state.
            reward (float): Immediate scalar reward obtained after
                executing ``action`` in ``observation``.
            next_observation (np.ndarray): Observation of the next state
                after the transition. Only used when ``terminated`` is
                ``False``.
            table (str): Either "A" or "B", indicating which Q-table
                to update (Q_A or Q_B).
            terminated (bool, optional): Flag indicating whether the
                transition ended the episode. If ``True``, no bootstrap
                term is added and the TD target is equal to ``reward``.

        Returns:
            None: The chosen Q-table is updated in-place.
        """
        if terminated:
            td_target = reward
        else:
            next_obs_index = self.model_a.observation_to_index(next_observation)
            max_next_action = (
                np.max(self.model_b.q_table[next_obs_index])
                if table == "A"
                else np.max(self.model_a.q_table[next_obs_index])
            )
            td_target = reward + self.gamma * max_next_action

        obs_index = self.model_a.observation_to_index(observation)

        if table == "A":
            td_error = td_target - self.model_a.q_table[obs_index][action]
            self.model_a.q_table[obs_index][action] += self.alpha * td_error
        else:
            td_error = td_target - self.model_b.q_table[obs_index][action]
            self.model_b.q_table[obs_index][action] += self.alpha * td_error

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
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
            reward = self.simulation()
            self.tb_writer.add_scalar("DoubleQLearning/Reward", reward, epoch)
            self.tb_writer.add_scalar("DoubleQLearning/Epsilon", self.epsilon, epoch)
            self.environment.reset()
            logger().info(
                f"Epoch {epoch + 1}/{epochs} completed with " f"epsilon {self.epsilon:.4f} and reward {reward}"
            )
        self.close_tb()

    def save_model(self) -> None:
        """
        Persist the Q-table to disk using NumPy binary format (.npy).
        """
        model = ModelQTable(observation_cardinality=self.model_a.observation_cardinality)
        model.q_table = self.model_a.q_table + self.model_b.q_table
        model.save(self.model_name)

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
        raise NotImplementedError("Method simulation() not implemented in TrainerDoubleQLearning.")
