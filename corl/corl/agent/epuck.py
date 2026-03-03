import logging
from collections import deque
from typing import Any

import numpy as np
import tensorflow as tf
from controller import Camera, DistanceSensor, Motor, Robot

import corl.utils.image as img
from corl.agent import Agent
from corl.utils.config import Config

CAMERA_FRAME_SIZE = 4  # Number of frames to stack for temporal observation
MAX_VELOCITY = 6.28  # Maximum wheel velocity in rad/s

logger = logging.getLogger(__name__)


class Epuck(Agent):
    """
    Base controller for the e-puck differential-drive robot in Webots.

    Extends :class:`~corl.agent.Agent` with e-puck-specific hardware:
    wheel motors, eight proximity sensors, and a front-facing camera.

    Sensors and camera are **not** initialised in ``__init__``; call
    :meth:`init_distance_sensors` and/or :meth:`init_camera` explicitly
    after construction to enable them.

    The discrete action space maps integer identifiers to
    ``(left_delta, right_delta)`` velocity pairs applied incrementally
    to the wheel motors each call to :meth:`act`.

    Attributes:
        motors: Left and right wheel :class:`Motor` instances.
        distance_sensors: List of eight :class:`DistanceSensor` instances,
            or ``None`` if not yet initialised.
        camera: Front :class:`Camera` instance, or ``None`` if not yet
            initialised.
        camera_frame_buffer: Circular buffer holding the last
            ``CAMERA_FRAME_SIZE`` preprocessed frames, or ``None`` if the
            camera has not been initialised.
        actions: Mapping from action integer to
            ``(left_delta, right_delta)`` velocity tuples.
    """

    motors: list[Motor] = []
    distance_sensors: list[DistanceSensor] | None = None
    camera: Camera | None = None
    camera_frame_buffer: deque | None = None

    def __init__(
        self,
        robot: Robot,
        timestep: int,
        action_repeat: int,
        config: Config,
        model: tf.keras.Model | np.ndarray | None = None,
    ):
        """
        Initialize the e-puck agent and its wheel motors.

        Calls the parent :meth:`~corl.agent.Agent.__init__`, initialises
        the wheel motors via :meth:`_init_motors`, and builds the action map.

        Distance sensors and camera are **not** enabled here; call
        :meth:`init_distance_sensors` and :meth:`init_camera` separately
        as needed.

        Args:
            robot: Webots ``Robot`` node to control.
            timestep: Simulation timestep in milliseconds.
            action_repeat: Number of consecutive simulation steps each
                selected action is held before a new one is requested.
            config: Application configuration forwarded to
                :class:`~corl.agent.Agent`.
            model: Optional pre-loaded model used in run mode.
        """
        super().__init__(
            robot=robot,
            timestep=timestep,
            action_repeat=action_repeat,
            config=config,
            model=model,
        )
        self._init_motors()

        self.actions = {
            0: (0.0, 0.0),  # Do nothing
            1: (0.1, 0.0),  # Turn right
            2: (0.0, 0.1),  # Turn left
            3: (0.1, 0.1),  # Move forward
            4: (-0.1, 0.0),  # Turn left backward
            5: (0.0, -0.1),  # Turn right backward
            6: (-0.1, -0.1),  # Move backward
            7: (0.1, -0.1),  # Quick right rotate
            8: (-0.1, 0.1),  # Quick left rotate
        }

    def _init_motors(self) -> None:
        """
        Initialize the e-puck's wheel motors for velocity control.

        Configures left and right wheel motors for continuous rotation by setting
        position to infinity and initial velocity to zero. This enables differential
        drive control via setVelocity().
        """
        self.motors = [
            self.robot.getDevice("left wheel motor"),
            self.robot.getDevice("right wheel motor"),
        ]
        self.motors[0].setPosition(float("inf"))
        self.motors[1].setPosition(float("inf"))
        self.motors[0].setVelocity(0.0)
        self.motors[1].setVelocity(0.0)
        logger.debug("Epuck motors initialized")

    def init_distance_sensors(self) -> None:
        """
        Initialize and enable the e-puck's eight proximity sensors.

        Sets up distance sensors (ps0–ps7) positioned around the robot's
        perimeter and enables them with the simulation timestep for
        continuous readings.

        Sensor layout:
            - ps0–ps2: Front-right quadrant
            - ps3–ps4: Rear quadrant
            - ps5–ps7: Front-left quadrant
        """

        self.distance_sensors = []
        names = ["ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7"]
        for i in range(8):
            self.distance_sensors.append(self.robot.getDevice(names[i]))
            self.distance_sensors[i].enable(self.timestep)
        logger.debug("Epuck distance sensors initialized")

    def init_camera(self) -> None:
        """
        Initialise the e-puck's front camera and frame buffer.

        Enables the camera device with the simulation timestep and creates
        a circular buffer capped at ``CAMERA_FRAME_SIZE`` preprocessed
        frames for temporal stacking (CNN input).
        """
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)
        self.camera_frame_buffer = deque(maxlen=CAMERA_FRAME_SIZE)
        logger.debug("Epuck camera initialized")

    def format_camera_image(self, observation: np.ndarray) -> np.ndarray:
        """
        Preprocess a raw camera image and append it to the frame buffer.

        Resizes ``observation`` to 42×42 pixels, converts it to grayscale,
        and normalises pixel values to ``[0, 1]``. The processed frame is
        appended to ``camera_frame_buffer`` and the last
        ``CAMERA_FRAME_SIZE`` frames are concatenated along the channel
        axis. A batch dimension is added before returning.

        Args:
            observation: Raw image array from ``camera.getImageArray()``.

        Returns:
            np.ndarray: Stacked frame array of shape
                ``(1, 42, 42, CAMERA_FRAME_SIZE)``.
        """
        frame = img.format_image(
            observation, shape=(42, 42), grayscale=True, normalize=True
        )
        self.camera_frame_buffer.append(frame)
        frame = img.concatenate_frames(self.camera_frame_buffer, CAMERA_FRAME_SIZE)
        frame = np.expand_dims(frame, axis=0)
        return frame

    def observe(self) -> dict[str, Any]:
        """
        Read active sensors and return the current observation payload.

        Only the subsystems that have been explicitly initialised are
        sampled:

        - ``"distance_sensors"`` (:class:`list` of ``float``) — present
          when :meth:`init_distance_sensors` has been called.
        - ``"camera"`` (raw image array) — present when
          :meth:`init_camera` has been called.

        Returns:
            dict[str, Any]: Observation dictionary with zero, one, or both of the
                keys above depending on which subsystems are active.
        """
        observation = {}
        if self.distance_sensors is not None:
            observation["distance_sensors"] = [
                s.getValue() for s in self.distance_sensors
            ]
        if self.camera is not None:
            observation["camera"] = self.camera.getImageArray()
        return observation

    def act(self, action: int) -> None:
        """
        Apply a velocity increment to both wheel motors.

        Looks up ``action`` in :attr:`actions` to retrieve a
        ``(left_delta, right_delta)`` pair and adds each delta to the
        corresponding motor's current velocity, clamped to the module-level
        ``MAX_VELOCITY`` constant.

        Args:
            action: Integer action identifier. Must be a key in
                :attr:`actions` (``0``–``8``).

        Raises:
            ValueError: If ``action`` is not a valid key in
                :attr:`actions`.
        """
        if action not in self.actions:
            raise ValueError(
                f"Invalid action {action}. Must be one of {list(self.actions)}."
            )

        velocity_motor_0 = self.motors[0].getVelocity()
        velocity_motor_1 = self.motors[1].getVelocity()

        if abs(velocity_motor_0 + self.actions[action][0]) < MAX_VELOCITY:
            self.motors[0].setVelocity(velocity_motor_0 + self.actions[action][0])

        if abs(velocity_motor_1 + self.actions[action][1]) < MAX_VELOCITY:
            self.motors[1].setVelocity(velocity_motor_1 + self.actions[action][1])

        logger.debug(
            f"Epuck action {action} executed: left velocity {self.motors[0].getVelocity()}, right velocity {self.motors[1].getVelocity()}"
        )
