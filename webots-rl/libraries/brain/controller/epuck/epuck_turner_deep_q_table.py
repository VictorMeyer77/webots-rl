"""
Deep Q-Learning Controller for E-puck Robot in Webots.

This module provides a controller for the e-puck robot that loads and executes
pre-trained Deep Q-Learning models (TensorFlow/Keras) for autonomous navigation.
The controller supports both camera-based and distance sensor-based observations
and uses the trained Q-network to select actions during inference.

The controller extends EpuckTurner and adds TensorFlow model loading and inference
capabilities for reinforcement learning policies.
"""

import os

import numpy as np
import tensorflow as tf
from brain.controller.epuck import EpuckTurner
from brain.model import MODEL_PATH, Model
from brain.utils.logger import logger
from brain.utils.register_tf import dueling_combine_streams
from controller import Robot


class EpuckTurnerDeepQTable(EpuckTurner):
    """
    E-puck controller that uses a Deep Q-Learning model for action selection.

    Attributes:
        model (tf.keras.models.Model | None): Loaded TensorFlow/Keras Q-network.
            None until load_model() is called.
    """

    model: tf.keras.models.Model | None

    def __init__(self, robot: Robot, timestep: int, max_speed: float):
        """
        Initialize the Deep Q-Learning controller.

        Sets up the controller with robot instance and prepares for model loading.
        The model is not loaded during initialization - it must be loaded separately
        using load_model().

        Args:
            robot (Robot): Webots robot instance from the controller script.
                Obtained via: robot = Robot()
            timestep (int): Simulation timestep in milliseconds.
                Should match supervisor timestep for synchronization.
                Typical value: 32 (31.25 Hz update rate)
            max_speed (float): Maximum angular velocity for wheel motors in rad/s.
                E-puck motor limit: ~6.28 rad/s (≈ 1 revolution/second)
        """

        super().__init__(robot=robot, timestep=timestep, max_speed=max_speed)
        self.model = None

    def set_model(self, model: Model) -> None:  # todo remove
        """
        Set model directly (not implemented - use load_model instead).

        This method is intentionally not implemented. Use load_model() to load
        pre-trained models from disk instead of setting models programmatically.

        Args:
            model (Model): Model instance (not used).

        Raises:
            NotImplementedError: Always raised. Use load_model() instead.
        """
        raise NotImplementedError("Used load_model instead")

    def load_model(self, name: str) -> None:
        """
        Load a pre-trained Deep Q-Learning model from disk.

        Loads a saved model in .keras format from the MODEL_PATH directory.
        The model is loaded with all its architecture, weights, and compilation
        configuration. After loading, the model is ready for inference.

        Args:
            name (str): Model name without file extension.
                The file {name}.keras must exist in MODEL_PATH.
                Example: "simple_arena_dueling_q_learning"

        File Format:
            - Format: Keras 3 .keras format (HDF5-based)
            - Location: MODEL_PATH/{name}.keras
            - Contains: architecture, weights, optimizer state, custom objects
        """
        path = os.path.join(MODEL_PATH, name + ".keras")
        self.model = tf.keras.models.load_model(
            path, custom_objects={"dueling_combine_streams": dueling_combine_streams}
        )
        self.model.summary()
        logger().info(f"Loaded TensorFlow model from {path}")

    def policy(self, observation: dict) -> int:
        """
        Select action using the loaded Q-network (greedy policy).

        Processes the observation (camera or distance sensors), formats it for
        the neural network, and returns the action with the highest Q-value.

        Args:
            observation (dict): Dictionary containing sensor data.
                - "camera": numpy array of camera image (if camera enabled)
                - "distance_sensors": list/array of distance values (if sensors enabled)

        Returns:
            int: Selected action index (argmax of Q-values).
        """
        if self.camera is not None:
            observation = np.array(observation["camera"]).astype(np.uint8)
            observation = self.format_camera_image(observation)
        elif self.distance_sensors is not None:
            observation = np.atleast_2d(np.array(observation["distance_sensors"]).astype(np.float32))
        else:
            raise RuntimeError("No valid observation source available.")

        outputs = self.model(observation)
        action = np.argmax(outputs)
        return action
