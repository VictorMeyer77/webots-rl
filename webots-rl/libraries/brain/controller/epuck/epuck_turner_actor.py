"""
E-puck Turner Actor Controller

This module provides an actor-based controller for e-puck robots that uses trained
neural network models to make navigation decisions. It extends the base EpuckTurner
class to enable policy-based control through TensorFlow models.

The actor supports both deterministic and stochastic action selection strategies,
making it suitable for deployment after reinforcement learning training (e.g., PPO).
"""

import os

import numpy as np
import tensorflow as tf
from brain.controller.epuck import EpuckTurner
from brain.model import MODEL_PATH, Model
from brain.utils.logger import logger
from controller import Robot


class EpuckTurnerActor(EpuckTurner):
    """
    Actor controller for e-puck robots using trained neural network policies.

    This class loads pre-trained TensorFlow models and executes learned policies
    to control robot navigation. It supports both camera-based and distance sensor-based
    observations, with configurable deterministic or stochastic action selection.

    Attributes:
        model (tf.keras.Model | None): The loaded TensorFlow Keras model used for
            action prediction. None until a model is loaded via load_model().
        stochastic (bool): Flag controlling action selection strategy. If True,
            actions are sampled probabilistically from the policy distribution.
            If False, the action with highest probability is selected deterministically.
    """

    model: tf.keras.Model | None
    stochastic: bool

    def __init__(self, robot: Robot, timestep: int, max_speed: float, stochastic: bool = False) -> None:

        super().__init__(robot=robot, timestep=timestep, max_speed=max_speed)
        self.model = None
        self.stochastic = stochastic

    def set_model(self, model: Model) -> None:
        """
        Placeholder method for setting a model instance.

        This method is intentionally not implemented. Use load_model() instead
        to load pre-trained models from disk.

        Args:
            model (Model): Model instance (not used).
        """
        raise NotImplementedError("Used load_model instead")

    def load_model(self, name: str) -> None:
        """
        Load a pre-trained TensorFlow model from the models directory.

        Loads a Keras model (.keras format) from the configured MODEL_PATH directory
        and assigns it to the model attribute for policy execution.

        Args:
            name (str): Model filename without the .keras extension.
        """
        path = os.path.join(MODEL_PATH, name + ".keras")
        self.model = tf.keras.models.load_model(path)
        logger().info(f"Loaded TensorFlow model from {path}")

    @staticmethod
    def _extract_logits(model_out):
        """
        Extract logits from model output, handling different output formats.

        Many reinforcement learning models return tuples containing both policy
        logits and value estimates. This method extracts only the logits (first element)
        when the output is a tuple or list, otherwise returns the output unchanged.

        Args:
            model_out: Model output, either a single tensor or tuple/list of tensors.

        Returns:
            Logits tensor for action selection.
        """
        if isinstance(model_out, (tuple, list)):
            return model_out[0]
        return model_out

    def policy(self, observation: dict) -> int:
        """
        Determine the next action based on current observations using the loaded model.

        This method processes observations (camera images or distance sensors), feeds them
        to the neural network model, and selects an action either deterministically
        (argmax) or stochastically (sampling from distribution) based on the stochastic flag.

        Args:
            observation (dict): Dictionary containing either:
                - "camera": Raw camera image data (uint8 array)
                - "distance_sensors": Distance sensor readings (float32 array)

        Returns:
            int: Selected action index.
        """
        if getattr(self, "camera", None) is not None:
            observation = np.array(observation["camera"], dtype=np.uint8)
            observation = self.format_camera_image(observation)
        elif getattr(self, "distance_sensors", None) is not None:
            observation = np.array(observation["distance_sensors"], dtype=np.float32)
            observation = np.atleast_2d(observation)
        else:
            raise RuntimeError("No valid observation source available.")

        x = tf.convert_to_tensor(observation, dtype=tf.float32)
        out = self.model(x, training=False)
        logits = self._extract_logits(out)

        if self.stochastic:
            action = tf.random.categorical(logits, 1)[0, 0].numpy()
            return int(action)

        return int(tf.argmax(logits, axis=-1)[0].numpy())

        # out = self.model(x, training=False)
        # logits = self._extract_logits(out)


#
# if self.stochastic:
#    # SAC: Sample from temperature-scaled softmax distribution
#    # Temperature is learned during training, baked into logits at inference
#    probs = tf.nn.softmax(logits, axis=-1)
#    action = tf.random.categorical(tf.math.log(probs), 1)[0, 0].numpy()
#    return int(action)
# else:
#    # Deterministic: Select action with highest probability
#    return int(tf.argmax(logits, axis=-1)[0].numpy())
