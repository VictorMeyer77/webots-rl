"""
Generic TensorFlow Model Controller for E-puck Robot in Webots.

This module provides a flexible, reusable controller for the e-puck robot that can load
and execute ANY pre-trained TensorFlow/Keras model for autonomous navigation. The controller
is model-agnostic and can be used with various neural network architectures including CNNs,
MLPs, RNNs, or custom models trained with different reinforcement learning algorithms.

Constants:
    FRAME_SIZE (int): Number of frames to stack for temporal context (default: 4).
        Set to 1 to disable frame stacking.
"""

import os
from collections import deque

import brain.utils.image as img
import numpy as np
import tensorflow as tf
from brain.controller.epuck import EpuckTurner
from brain.model import MODEL_PATH, Model
from brain.utils.logger import logger
from controller import Robot

FRAME_SIZE = 4


class EpuckTurnerTensorflow(EpuckTurner):
    """
    Generic TensorFlow model controller for e-puck robot autonomous navigation.

    A flexible, model-agnostic controller that can load and execute ANY pre-trained
    TensorFlow/Keras model for robot control. Supports various architectures (CNNs, MLPs,
    RNNs) and training algorithms (DQN, PPO, A3C, etc.) with minimal configuration.

    The controller handles:
    - Model loading from .keras files
    - Observation preprocessing (customizable)
    - Frame stacking for temporal context
    - Action selection via model inference
    - Supervisor-controller communication protocol

    Key Features:
        - **Model Agnostic**: Works with any TensorFlow model architecture
        - **Flexible Input**: Supports vision, sensors, or hybrid observations
        - **Pure Inference**: Deployment-focused, no training/exploration
        - **Frame Stacking**: Configurable temporal window (default: 4 frames)
        - **Easy Integration**: Simple load-and-run API

    Attributes:
        model (tf.keras.models.Sequential | None): Loaded TensorFlow model for inference.
            None until load_model() is called.
            Expected to output action values (Q-values) or probabilities.
        frame_buffer (deque): Circular buffer storing recent processed frames.
            Size: FRAME_SIZE (default: 4)
            Used for temporal context in sequential decision making.

    Inherited Attributes:
        robot (Robot): Webots robot instance
        timestep (int): Simulation timestep in milliseconds
        max_speed (float): Maximum wheel velocity in rad/s
        motors (list[Motor]): Left and right wheel motors
        camera (Camera): Front-facing camera (if initialized)
        distance_sensors (list[DistanceSensor]): 8 distance sensors (if initialized)
        queue (MessageQueue): Communication queue with supervisor
    """

    model: tf.keras.models.Sequential | None
    frame_buffer: deque

    def __init__(self, robot: Robot, timestep: int, max_speed: float):
        """
        Initialize the TensorFlow-based e-puck controller.

        Sets up the controller with robot instance, initializes an empty frame buffer
        for temporal context, and prepares for model loading. The model is not loaded
        during initialization - it must be loaded separately using load_model().

        Args:
            robot (Robot): Webots robot instance from the controller script.
                Obtained via: robot = Robot()
            timestep (int): Simulation timestep in milliseconds.
                Should match supervisor timestep for synchronization.
                Typical value: 32 (31.25 Hz update rate)
            max_speed (float): Maximum angular velocity for wheel motors in rad/s.
                E-puck motor limit: ~6.28 rad/s (≈ 1 revolution/second)
                Higher values may not be achievable by physical robot

        Attributes Initialized:
            - self.frame_buffer: Empty deque with maxlen=FRAME_SIZE
            - self.model: Set to None (must call load_model() before use)
        """

        super().__init__(robot=robot, timestep=timestep, max_speed=max_speed)
        self.frame_buffer = deque(maxlen=FRAME_SIZE)
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
        Load a pre-trained TensorFlow/Keras model from disk.

        Loads a saved model in .keras format from the MODEL_PATH directory.
        The model is loaded with all its architecture, weights, optimizer state,
        and compilation configuration. After loading, the model is ready for
        inference without any additional setup.

        Args:
            name (str): Model name without file extension.
                The file {name}.keras must exist in MODEL_PATH.
                Example: "simple_arena_dqn" loads "simple_arena_dqn.keras"

        File Format:
            Expected format: Keras 3 .keras format (HDF5-based)
            Location: MODEL_PATH/{name}.keras

            The .keras file contains:
            - Model architecture (layers, connections)
            - Trained weights (all parameters)
            - Optimizer state (Adam parameters, etc.)
            - Loss and metrics configuration
        """

        path = os.path.join(MODEL_PATH, name + ".keras")
        self.model = tf.keras.models.load_model(path)
        self.model.summary()
        logger().info(f"Loaded TensorFlow model from {path}")

    def policy(self, observation: dict) -> int:
        """
        Select action using loaded TensorFlow model (greedy policy).

        Processes raw camera observation through the complete vision pipeline
        (resize → grayscale → normalize → stack frames) and performs CNN inference
        to select the best action. This is a pure exploitation strategy (no exploration)
        suitable for deployment and evaluation of trained models.

        Observation Processing Pipeline:
            1. **Extract**: Get camera array from observation dict
            2. **Convert**: Cast to uint8 [0-255] (from Webots format)
            3. **Resize**: Scale to (42, 42) for CNN input
            4. **Grayscale**: Convert RGB to single channel
            5. **Normalize**: Scale to [0.0, 1.0] range
            6. **Buffer**: Append to frame_buffer (circular deque)
            7. **Stack**: Concatenate last 4 frames → (42, 42, 4)
            8. **Batch**: Add dimension → (1, 42, 42, 4)
            9. **Inference**: CNN forward pass → Q-values
            10. **Select**: Argmax to choose best action

        Args:
            observation (dict): Dictionary containing sensor readings.
                Required key: "camera" - RGB image array from Webots camera
                Expected shape: (height, width, 3) with uint8 values
                Optional keys: "distance_sensors" (unused in default implementation)

        Returns:
            int: Selected action index (0-3).
                - 0: Move forward
                - 1: Turn left
                - 2: Turn right
                - 3: Move backward

                Action is chosen by argmax over predicted Q-values (greedy policy).
        """
        observation = np.array(observation["camera"]).astype(np.uint8)
        frame = img.format_image(observation, shape=(42, 42), grayscale=True, normalize=True)
        self.frame_buffer.append(frame)
        frame = img.concatenate_frames(self.frame_buffer, FRAME_SIZE)
        frame = np.expand_dims(frame, axis=0)

        action_values = self.model.predict(frame, verbose=0)
        logger().debug(f"Model predictions (Q-values): {action_values[0]}")

        action = np.argmax(action_values)
        logger().info(f"Selected action: {action} with Q-value: {action_values[0][action]:.4f}")

        return int(action)

    def train(self) -> None:
        """Execute the controller-side training communication loop.

        Sequence per iteration:
          1. Synchronize: wait for ``{'sync': 1}`` then reply ``{'ack': 1}``
             once.
          2. Observation: send discretized sensor data.
          3. Action reception: wait for supervisor-selected action.
          4. Actuation: apply action via :meth:`act` and notify completion
             with ``{'step': step_index}``.
          5. Repeat until simulation ends (``robot.step()`` returns ``-1``).

        Notes:
          * Exploration (epsilon-greedy) is handled supervisor-side; the
            controller always executes the received actions.
          * The controller does not update the Q-table; it only supplies
            observations and executes actions.
        """
        step_index = 0
        sync = False
        step_observation = None
        step_action = None

        while self.robot.step(self.timestep) != -1:

            self.queue.clear_buffer()

            # (1) Initial synchronization handshake on the very first step.
            if not sync:
                if not self.queue.search_message("sync"):
                    continue
                else:
                    self.queue.send({"ack": 1})
                    sync = True
                    logger().debug("Synchronization with supervisor successful.")

            # (2) Send observation.
            if step_observation is None:
                step_observation = self.observe()
                self.queue.send({"observation": step_observation})

            # (3) Await action.
            if step_action is None:
                action_messages = self.queue.search_message("action")
                if not action_messages:
                    continue
                else:
                    step_action = action_messages[0]["action"]
                    logger().debug(f"Received action {step_action}")

            # (4) Execute action and notify step completion.
            self.act(step_action)
            self.queue.send({"step": step_index})

            # (5) Prepare for next iteration.
            step_index += 1
            step_observation = None
            step_action = None
            logger().debug(f"Controller completed step {step_index}")
