"""
Tabular Q-table model utilities.

Provides a persistable NumPy-backed Q-table plus an observation indexing
scheme for discrete, factorized sensor readings. Each observation component
is assumed to have identical discrete cardinality ``observation_cardinality``,
allowing mixed-radix encoding into a single linear state index.

Notes:
  * ``observation_to_index`` performs O(n) mixed-radix accumulation.
  * Caller must allocate ``q_table`` before saving.
  * This class is agnostic to the learning algorithm; update logic
    (Q-learning, SARSA, etc.) resides in the trainer classes.
"""

import os

import numpy as np
from brain.model import MODEL_PATH, Model
from brain.utils.logger import logger


class ModelQTable(Model):
    """Persistable tabular Q-table for discrete RL algorithms.

    This model stores a 2-D NumPy array of Q-values indexed by a linearized
    state index and an action index. It is intended to be used with
    tabular algorithms such as Q-learning or SARSA.

    Attributes:
        observation_cardinality (int): Number of discrete bins per
            observation feature.
        q_table (np.ndarray | None): Array of shape
            ``(state_space_size, action_space_size)``; must be allocated
            externally before :meth:`save` is called.
    """

    observation_cardinality: int
    q_table: np.ndarray | None

    def __init__(self, observation_cardinality: int):
        """Create a Q-table model without allocating the table itself.

        Args:
            observation_cardinality (int): Uniform discrete bin count per
                observation component. Used by :meth:`observation_to_index`
                to encode multi-dimensional observations into a single
                integer index.
        """
        self.observation_cardinality = observation_cardinality
        self.q_table = None

    def load(self, name: str) -> None:
        """
        Load a persisted Q-table from disk.

        Args:
            name (str): Base filename (without extension) located under `MODEL_PATH`.
        """
        path = os.path.join(MODEL_PATH, name + ".npy")
        self.q_table = np.load(path)
        logger().info(f"Model loaded successfully from {path}")

    def save(self, name: str) -> None:
        """
        Persist the current Q-table to disk.

        Args:
            name (str): Base filename (without extension) for the output `.npy`.

        Raises:
            RuntimeError: If `q_table` is unset (`None`).
        """
        if self.q_table is None:
            raise RuntimeError("Cannot save: q_table is None.")
        model_path = os.path.join(MODEL_PATH, name + ".npy")
        np.save(model_path, self.q_table)
        logger().info(f"Model saved successfully at {model_path}")

    def observation_to_index(self, observation: np.ndarray) -> int:
        """
        Encode a discrete observation vector into a linear table index.

        Performs mixed-radix accumulation assuming each element has radix
        `observation_cardinality`:
            index = sum( obs[i] * (observation_cardinality ** i) )

        Args:
            observation (np.ndarray): 1-D array of integer bin values.

        Returns:
            int: Linear index suitable for addressing `q_table`.
        """
        index = 0
        mul = 1
        for obs in observation:
            index += obs * mul
            mul *= self.observation_cardinality
        return index
