import logging
from typing import Any

import numpy as np
from controller import DistanceSensor, Motor, Robot

from corl.agent.agent import Agent
from corl.agent.camera import Camera
from corl.model.model import Model

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
        motors: Left and right wheel :class:`Motor` instances, in that order.
        distance_sensors: List of eight :class:`DistanceSensor` instances
            (``ps0``–``ps7``), or ``None`` if not yet initialised.
        camera: Front-facing :class:`~corl.agent.camera.Camera` wrapper
            instance, or ``None`` if not yet initialised.
        actions: Mapping from action integer (``0``–``8``) to
            ``(left_delta, right_delta)`` velocity increment tuples.
    """

    motors: list[Motor] = []
    distance_sensors: list[DistanceSensor] | None = None
    camera: Camera | None = None
    actions: dict[int, tuple[float, float]]

    def __init__(
        self,
        robot: Robot,
        timestep: int,
        action_repeat: int,
        model: Model | None,
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
            model: Optional pre-loaded model used by :meth:`~corl.agent.Agent.policy`.
        """
        super().__init__(
            robot=robot,
            timestep=timestep,
            action_repeat=action_repeat,
            model=model,
        )
        self._init_motors()

        self.actions = {
            0: (0.0, 0.0),  # Do nothing
            1: (0.1, 0.0),  # Turn left
            2: (0.0, 0.1),  # Turn right
            3: (0.1, 0.1),  # Move forward
            4: (-0.1, 0.0),  # Turn right backward
            5: (0.0, -0.1),  # Turn left backward
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
        for name in names:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.timestep)
            self.distance_sensors.append(sensor)
        logger.debug("Epuck distance sensors initialized")

    def init_camera(
        self,
        image_shape: tuple[int, int] | None = None,
        grayscale: bool = True,
        normalize: bool = True,
    ) -> None:
        """
        Initialise the e-puck's front camera with preprocessing settings.

        Wraps the Webots ``"camera"`` device in a
        :class:`~corl.agent.camera.Camera` instance configured for temporal
        frame stacking. The buffer holds ``CAMERA_FRAME_SIZE`` frames and is
        zero-padded at the start of each episode until full.

        Args:
            image_shape: Target ``(height, width)`` to resize each frame to,
                or ``None`` to keep the native camera resolution.
                Defaults to ``None``.
            grayscale: Convert BGRA frames to grayscale when ``True``.
                Defaults to ``True``.
            normalize: Scale pixel values to ``[0.0, 1.0]`` when ``True``.
                Defaults to ``True``.
        """
        self.camera = Camera(
            self.robot.getDevice("camera"),
            self.timestep,
            frame_size=CAMERA_FRAME_SIZE,
            image_shape=image_shape,
            grayscale=grayscale,
            normalize=normalize,
        )

    def observe(self) -> dict[str, Any]:
        """
        Read active sensors and return the current observation payload.

        Only the subsystems that have been explicitly initialised are
        sampled:

        - ``"distance_sensors"`` (``list[float]``) — present when
          :meth:`init_distance_sensors` has been called. Contains one
          float per sensor (``ps0``–``ps7``).
        - ``"camera"`` (``list``) — present when :meth:`init_camera` has
          been called. Nested list representation of a stacked-frame array
          with shape ``(height, width, CAMERA_FRAME_SIZE)`` for grayscale,
          or ``(height, width, channels * CAMERA_FRAME_SIZE)`` for colour,
          where ``height`` and ``width`` match the configured
          ``image_shape`` (or native camera resolution if unset).

        Returns:
            dict[str, Any]: Observation dictionary with zero, one, or both
            keys above, depending on which subsystems are active.
        """
        observation = {}
        if self.distance_sensors is not None:
            observation["distance_sensors"] = [
                s.getValue() for s in self.distance_sensors
            ]
        if self.camera is not None:
            observation["camera"] = self.camera.process_camera_image().tolist()
        return observation

    def act(self, action: list[float]) -> None:
        """
        Apply a velocity increment to both wheel motors.

        Looks up the action index in :attr:`actions` to retrieve a
        ``(left_delta, right_delta)`` pair and adds each delta to the
        corresponding motor's current velocity. The update is only applied
        when the resulting speed stays within ``±MAX_VELOCITY``; if the
        clamp would be exceeded the motor velocity is left unchanged for
        that wheel.

        Args:
            action: Single-element list containing the integer action index
                as a float (e.g. ``[2.0]``). The index must be a key in
                :attr:`actions` (``0``–``8``).

        Raises:
            ValueError: If the action index is not a valid key in :attr:`actions`.
        """
        action_index = int(action[0])
        if action_index not in self.actions:
            raise ValueError(
                f"Invalid action {action_index}. Must be one of {list(self.actions)}."
            )

        left_delta, right_delta = self.actions[action_index]

        new_left = np.clip(
            self.motors[0].getVelocity() + left_delta, -MAX_VELOCITY, MAX_VELOCITY
        )
        new_right = np.clip(
            self.motors[1].getVelocity() + right_delta, -MAX_VELOCITY, MAX_VELOCITY
        )

        self.motors[0].setVelocity(new_left)
        self.motors[1].setVelocity(new_right)

        logger.debug(
            f"Epuck action {action_index} executed: left velocity {self.motors[0].getVelocity()}, right velocity {self.motors[1].getVelocity()}"
        )
