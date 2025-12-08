"""
This module provides base classes for controlling an e-puck robot in Webots.

Constants:
    FRAME_SIZE: Number of frames stacked for temporal observation (default: 4).

Classes:
    - BaseEpuck: Initializes distance sensors, camera, and motors. Provides observation,
                 reset, and training communication methods.
    - EpuckTurner: Implements discrete movement actions (forward, left, right, backward)
                   for the e-puck robot.

These classes are designed for extensibility and integration with reinforcement learning
training or simulation environments.
"""

from collections import deque

import brain.utils.image as img
import numpy as np
from brain.controller import BaseController
from brain.utils.logger import logger
from controller import Camera, DistanceSensor, Motor, Robot

FRAME_SIZE = 4


class BaseEpuck(BaseController):
    """
    Base controller for the e-puck robot in Webots.

    Provides core functionality for sensor management, observation collection,
    and training communication. Supports both distance sensors and camera-based
    observations for reinforcement learning applications.

    Attributes:
        max_speed (float): Maximum angular velocity for wheel motors in rad/s.
        motors (list[Motor]): Left and right wheel motor devices [left, right].
        distance_sensors (list[DistanceSensor] | None): Eight proximity sensors (ps0-ps7).
            None until init_distance_sensors() is called.
        camera (Camera | None): Camera device for visual observations.
            None until init_camera() is called.
        frame_buffer (deque | None): Circular buffer storing last FRAME_SIZE frames.
            None until init_camera() is called. Used for temporal frame stacking.
    """

    max_speed: float
    motors: list[Motor]
    distance_sensors: list[DistanceSensor] | None
    camera: Camera | None
    frame_buffer: deque | None

    def __init__(self, robot: Robot, timestep: int, max_speed: float):
        """
        Initialize the BaseEpuck controller.

        Sets up motors for velocity control and prepares sensor attributes.
        Sensors must be explicitly enabled via init_distance_sensors() or
        init_camera() before use.

        Args:
            robot (Robot): Webots robot instance from the controller script.
                Obtained via: robot = Robot()
            timestep (int): Simulation timestep in milliseconds.
                Should match supervisor timestep for synchronization.
                Typical value: 32 (31.25 Hz update rate)
            max_speed (float): Maximum angular velocity for wheel motors in rad/s.
                E-puck motor limit: ~6.28 rad/s (≈ 1 revolution/second)

        Attributes Initialized:
            - self.motors: Left and right wheel motors set to velocity mode
            - self.distance_sensors: Set to None (call init_distance_sensors() to enable)
            - self.camera: Set to None (call init_camera() to enable)
            - self.frame_buffer: Set to None (created when camera is initialized)
        """
        super().__init__(robot=robot, timestep=timestep)
        self.max_speed = max_speed
        self._init_motors()
        self.distance_sensors = None
        self.camera = None

    def _init_motors(self) -> None:
        """
        Initialize the e-puck's wheel motors for velocity control.

        Configures left and right wheel motors for continuous rotation by setting
        position to infinity and initial velocity to zero. This enables differential
        drive control via setVelocity().
        """
        self.motors = [self.robot.getDevice("left wheel motor"), self.robot.getDevice("right wheel motor")]
        self.motors[0].setPosition(float("inf"))
        self.motors[1].setPosition(float("inf"))
        self.motors[0].setVelocity(0.0)
        self.motors[1].setVelocity(0.0)
        logger().debug("Epuck motors initialized")

    def init_distance_sensors(self) -> None:
        """
        Initialize and enable the e-puck's eight proximity sensors.

        Sets up distance sensors (ps0-ps7) positioned around the robot's perimeter
        and enables them with the simulation timestep for continuous readings.

        Sensor Layout:
            - ps0-ps2: Front-right quadrant
            - ps3-ps4: Rear quadrant
            - ps5-ps7: Front-left quadrant
        """

        self.distance_sensors = []
        names = ["ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7"]
        for i in range(8):
            self.distance_sensors.append(self.robot.getDevice(names[i]))
            self.distance_sensors[i].enable(self.timestep)
        logger().debug("Epuck distance sensors initialized")

    def init_camera(self) -> None:
        """
        Initialize the e-puck's camera and frame buffer.

        Enables the camera device and creates a circular buffer for temporal frame
        stacking. The buffer maintains the last FRAME_SIZE frames for CNN input.
        """
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)
        self.frame_buffer = deque(maxlen=FRAME_SIZE)
        logger().debug("Epuck camera initialized")

    def observe(self) -> dict:
        """
        Collect current sensor and camera observations from the e-puck robot.

        Gathers all enabled sensor data for reinforcement learning or control.
        Only includes data from initialized sensors (distance_sensors or camera).

        Returns:
            dict: Observation dictionary with the following possible keys:
                - 'distance_sensors' (list[float]): Eight proximity sensor values (ps0-ps7).
                    Range: [0, 4095] where higher values indicate closer objects.
                    Only present if init_distance_sensors() was called.
                - 'camera' (list[list[list[int]]]): RGB camera image array.
                    Shape: (height, width, 3) with uint8 values [0-255].
                    Typical e-puck resolution: (52, 39, 3).
                    Only present if init_camera() was called.
        """
        observations = {}
        if self.distance_sensors is not None:
            observations["distance_sensors"] = [s.getValue() for s in self.distance_sensors]
        if self.camera is not None:
            observations["camera"] = self.camera.getImageArray()
        logger().debug(f"Epuck readings: {observations.keys()}")
        return observations

    def format_camera_image(self, observation: np.ndarray) -> np.ndarray:
        """
        Process raw camera image for CNN input with temporal frame stacking.

        Transforms a single camera observation into a stacked multi-frame tensor
        suitable for CNN inference. This pipeline ensures consistent preprocessing
        between training and deployment, and provides temporal context for better
        decision making.

        Processing Steps:
            1. **Resize & Normalize**: Scale to (42, 42), convert to grayscale, normalize to [0, 1]
            2. **Buffer Update**: Append processed frame to circular frame_buffer (FIFO)
            3. **Frame Stacking**: Concatenate last FRAME_SIZE frames along channel axis
            4. **Batch Dimension**: Add dimension for batch inference (single sample)

        Args:
            observation (np.ndarray): Raw camera image from Webots.
                Expected shape: (height, width, 3) with uint8 values [0-255]
                Typically (52, 39, 3) from e-puck camera

        Returns:
            np.ndarray: Preprocessed stacked frames ready for CNN inference.
                Shape: (1, 42, 42, FRAME_SIZE)
                - Dimension 0: Batch size (always 1 for single inference)
                - Dimensions 1-2: Spatial dimensions (42x42 pixels)
                - Dimension 3: Temporal stack (FRAME_SIZE=4 grayscale frames)
                dtype: float32, normalized to [0.0, 1.0]
        """
        frame = img.format_image(observation, shape=(42, 42), grayscale=True, normalize=True)
        self.frame_buffer.append(frame)
        frame = img.concatenate_frames(self.frame_buffer, FRAME_SIZE)
        frame = np.expand_dims(frame, axis=0)
        return frame

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


class EpuckTurner(BaseEpuck):
    """
    E-puck controller with discrete differential drive actions.

    Extends BaseEpuck to implement four discrete movement primitives using
    differential drive control. Suitable for grid-like or discrete action
    space reinforcement learning tasks.

    Actions:
        - 0: Forward (both wheels at 50% max_speed)
        - 1: Turn left (left wheel backward, right forward)
        - 2: Turn right (left wheel forward, right backward)
        - 3: Backward (both wheels at -50% max_speed)

    Attributes:
        actions (dict[int, list[float]]): Mapping from action index to [left_speed, right_speed].
            Each speed is ±0.5 × max_speed.
    """

    def __init__(self, robot: Robot, timestep: int, max_speed: float):
        """
        Initialize the EpuckTurner controller.

        Sets up discrete action space with four movement primitives based on
        differential drive control at 50% of maximum speed.

        Args:
            robot (Robot): Webots robot instance from the controller script.
            timestep (int): Simulation timestep in milliseconds.
            max_speed (float): Maximum angular velocity for wheel motors in rad/s.
        """
        super().__init__(robot=robot, timestep=timestep, max_speed=max_speed)
        self.actions = {
            0: [0.5 * self.max_speed, 0.5 * self.max_speed],  # forward
            1: [-0.5 * self.max_speed, 0.5 * self.max_speed],  # left
            2: [0.5 * self.max_speed, -0.5 * self.max_speed],  # right
            3: [-0.5 * self.max_speed, -0.5 * self.max_speed],  # backward
        }

    def act(self, action: int) -> None:
        """
        Execute a discrete movement action by setting wheel velocities.

        Applies differential drive velocities corresponding to the action index.
        Wheel velocities are set at ±50% of max_speed for forward/backward/turning
        movements.

        Args:
            action (int): Discrete action index (0-3).
                - 0: Move forward
                - 1: Turn left (rotate counter-clockwise)
                - 2: Turn right (rotate clockwise)
                - 3: Move backward
        """

        if action not in self.actions:
            raise ValueError("Action must be an integer between 0 and 3.")

        logger().debug(
            f"Epuck motor velocities updated with left_speed = {self.actions[action][0]} "
            f"and right_speed = {self.actions[action][1]}"
        )

        self.motors[0].setVelocity(self.actions[action][0])
        self.motors[1].setVelocity(self.actions[action][1])
