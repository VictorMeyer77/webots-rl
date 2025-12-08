"""
REINFORCE Policy Gradient Controller for E-puck Robot in Webots.

This module provides a controller for the e-puck robot that loads and executes
pre-trained REINFORCE (policy gradient) models for autonomous navigation. Unlike
value-based methods (Q-learning), this controller uses a stochastic policy network
that outputs action probabilities.

The controller extends EpuckTurner and adds TensorFlow model loading and policy
inference capabilities for reinforcement learning with policy gradient methods.

Key Differences from Q-Learning:
    - Outputs action probabilities (via softmax) instead of Q-values
    - Uses stochastic policy during training (exploration via probability distribution)
    - Uses greedy policy during inference (argmax of probabilities)
    - No value function estimation required
"""

import os

import numpy as np
import tensorflow as tf
from brain.controller.epuck import EpuckTurner
from brain.model import MODEL_PATH, Model
from brain.utils.logger import logger
from brain.utils.register_tf import dueling_combine_streams
from controller import Robot


class EpuckTurnerReinforce(EpuckTurner):
    """
    E-puck controller that uses a REINFORCE policy network for action selection.

    This controller loads a trained policy gradient model and uses it to select
    actions during autonomous navigation. The policy network outputs action logits
    that are converted to probabilities via softmax, and the action with highest
    probability is selected (greedy inference).

    Attributes:
        model (tf.keras.models.Model | None): Loaded TensorFlow/Keras policy network.
            None until load_model() is called. Network outputs action logits.
    """

    model: tf.keras.models.Model | None

    def __init__(self, robot: Robot, timestep: int, max_speed: float):
        """
        Initialize the REINFORCE policy gradient controller.

        Sets up the controller with robot instance and prepares for model loading.
        The policy model is not loaded during initialization - it must be loaded
        separately using load_model().

        Args:
            robot (Robot): Webots robot instance from the controller script.
                Obtained via: robot = Robot()
            timestep (int): Simulation timestep in milliseconds.
                Should match supervisor timestep for synchronization.
                Typical value: 64 (15.625 Hz update rate)
            max_speed (float): Maximum angular velocity for wheel motors in rad/s.
                E-puck motor limit: ~6.28 rad/s (≈ 1 revolution/second)
        """
        super().__init__(robot=robot, timestep=timestep, max_speed=max_speed)
        self.model = None

    def set_model(self, model: Model) -> None:
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
        Load a pre-trained REINFORCE policy network from disk.

        Loads a saved model in .keras format from the MODEL_PATH directory.
        The model is loaded with all its architecture, weights, optimizer state,
        and compilation configuration. After loading, the model is ready for
        inference without any additional setup.

        Args:
            name (str): Model name without file extension.
                The file {name}.keras must exist in MODEL_PATH.
                Example: "simple_arena_reinforce_8zAT"

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
        Select action using the loaded policy network (greedy inference).

        Processes the observation (camera or distance sensors), formats it for
        the neural network, converts output logits to probabilities via softmax,
        and returns the action with the highest probability.

        Args:
            observation (dict): Dictionary containing sensor data.
                - "camera": numpy array of camera image (if camera enabled)
                - "distance_sensors": list/array of distance values (if sensors enabled)

        Returns:
            int: Selected action index (argmax of action probabilities).
                For simple_arena: 0=forward, 1=left, 2=right

        Raises:
            RuntimeError: If neither camera nor distance sensors are available.

        Processing Pipeline:
            1. **Observation Processing**:
               - Camera: Convert to uint8, format via format_camera_image()
               - Distance sensors: Convert to float32, reshape to 2D array
            2. **Network Inference**: Pass observation through policy network
            3. **Softmax Conversion**: Convert logits to probabilities [0, 1]
            4. **Action Selection**: Return action with highest probability (greedy)
        """
        if self.camera is not None:
            observation = np.array(observation["camera"]).astype(np.uint8)
            observation = self.format_camera_image(observation)
        elif self.distance_sensors is not None:
            observation = np.atleast_2d(np.array(observation["distance_sensors"]).astype(np.float32))
        else:
            raise RuntimeError("No valid observation source available.")

        outputs = self.model(observation)
        outputs = tf.nn.softmax(outputs[0]).numpy()
        action = np.argmax(outputs)
        return action
