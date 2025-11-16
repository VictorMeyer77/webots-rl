"""
Monte Carlo Q-table model utilities.

Provides a concrete `ModelMonteCarlo` implementation that:
- Loads and saves NumPy `.npy` Q-table files located under `MODEL_PATH`.
- Converts discrete sensor observations into linear indices via mixed-radix encoding.


Notes:
- `MODEL_PATH` imported from `brain.model`; consider parameterizing for portability.
- `observation_to_index` assumes each observation element is in `[0, observation_cardinality - 1]`.
"""

import os

import numpy as np
from brain.model import MODEL_PATH, Model
from brain.utils.logger import logger


class ModelMonteCarlo(Model):
    """
    Persistable Monte Carlo Q-table backed by NumPy arrays.

    Attributes:
        observation_cardinality (int): Number of discrete bins per sensor.
        q_table (np.ndarray | None): Loaded Q-table of shape (state_space, action_count) or None if not loaded.
    """

    observation_cardinality: int
    q_table: np.ndarray | None

    def __init__(self, observation_cardinality: int):
        """
        Initialize the model without loading data.

        Args:
            observation_cardinality (int): Sensor bin count used for index encoding.
        """
        self.observation_cardinality = observation_cardinality
        self.q_table = None

    def load(self, name: str) -> None:
        """
        Load a Q-table `.npy` file into memory.

        Args:
            name (str): Base filename (without extension) under `MODEL_PATH`.

        Raises:
            FileNotFoundError: If the target file does not exist.
            ValueError: If loaded object is not a NumPy array.
        """
        path = os.path.join(MODEL_PATH, name + ".npy")
        self.q_table = np.load(path)
        logger().info(f"Model loaded successfully from {path}")

    def save(self, name: str) -> None:
        """
        Persist the Q-table to disk using NumPy binary format (`.npy`).

        Args:
            name (str): Base filename (without extension) for output.

        Raises:
            RuntimeError: If `q_table` is None.
        """
        if self.q_table is None:
            raise RuntimeError("Cannot save: q_table is None.")
        model_path = os.path.join(MODEL_PATH, name + ".npy")
        np.save(model_path, self.q_table)
        logger().info(f"Model saved successfully at {model_path}")

    def observation_to_index(self, observation: np.ndarray) -> int:
        """
        Convert a list of discrete sensor values into a single linear index.

        Mixed-radix accumulation where each position has radix = `observation_cardinality`.

        Args:
            observation (ndarray): Discrete sensor bins.

        Returns:
            int: Linear index usable to address the Q-table.
        """
        index = 0
        mul = 1
        for obs in observation:
            index += obs * mul
            mul *= self.observation_cardinality
        return index
