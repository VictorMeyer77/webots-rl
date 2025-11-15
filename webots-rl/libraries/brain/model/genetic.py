"""
Genetic model persistence utilities.

This module defines `ModelGenetic`, a lightweight container for an evolved
action vector (genome) produced by a genetic algorithm trainer. It provides
simple `load` and `save` methods that serialize the genome as a NumPy
`.npy` file inside the configured `MODEL_PATH`.
"""

import os

import numpy as np
from brain.model import MODEL_PATH, Model
from brain.utils.logger import logger


class ModelGenetic(Model):
    """
    Concrete model holding a genetic algorithm action vector.

    Attributes:
        actions: NumPy array representing the best evolved genome. `None`
            until set by training or `load()`.
    """

    actions: np.ndarray | None

    def load(self, name: str) -> None:
        """
        Load a saved genome from disk.

        Args:
            name: Base file name (without extension) of the stored model.
        """
        path = os.path.join(MODEL_PATH, name + ".npy")
        self.actions = np.load(path)
        logger().info(f"Model loaded successfully from {path}")

    def save(self, name: str) -> None:
        """
        Persist the current genome to disk as `.npy`.

        Args:
            name: Base file name (without extension) to write.
        """
        model_path = os.path.join(MODEL_PATH, name + ".npy")
        np.save(model_path, self.actions)
        logger().info(f"Model saved successfully at {model_path}")
