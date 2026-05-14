import logging
from typing import Any

from controller import DistanceSensor, Motor, Robot

from corl.agent.agent import Agent
from corl.agent.camera import Camera
from corl.model.model import Model

CAMERA_FRAME_SIZE = 4  # Number of frames to stack for temporal observation
MAX_VELOCITY = 6.28  # Maximum wheel velocity in rad/s

logger = logging.getLogger(__name__)


class EpuckBase(Agent):
    """
    Hardware base class for the e-puck differential-drive robot in Webots.

    Manages wheel motors, eight proximity sensors, and a front-facing camera.
    Does **not** define an action space — subclasses implement :meth:`act`
    for their respective space (discrete or continuous).

    Sensors and camera are **not** initialised in ``__init__``; call
    :meth:`init_distance_sensors` and/or :meth:`init_camera` explicitly
    after construction to enable them.

    Attributes:
        motors: Left and right wheel :class:`Motor` instances, in that order.
        distance_sensors: List of eight :class:`DistanceSensor` instances
            (``ps0``–``ps7``), or ``None`` if not yet initialised.
        camera: Front-facing :class:`~corl.agent.camera.Camera` wrapper
            instance, or ``None`` if not yet initialised.
    """

    motors: list[Motor] = []
    distance_sensors: list[DistanceSensor] | None = None
    camera: Camera | None = None

    def __init__(
        self,
        robot: Robot,
        timestep: int,
        action_repeat: int,
        model: Model | None,
    ):
        """
        Initialize the e-puck base agent and its wheel motors.

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

    def _init_motors(self) -> None:
        """
        Initialize the e-puck's wheel motors for velocity control.

        Configures left and right wheel motors for continuous rotation by setting
        position to infinity and initial velocity to zero.
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
        Initialize and enable the e-puck's eight proximity sensors (ps0–ps7).

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

        Args:
            image_shape: Target ``(height, width)`` to resize each frame to,
                or ``None`` to keep the native camera resolution.
            grayscale: Convert BGRA frames to grayscale when ``True``.
            normalize: Scale pixel values to ``[0.0, 1.0]`` when ``True``.
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

        Keys present depend on which subsystems have been initialised:

        - ``"distance_sensors"`` — ``list[float]``, one value per sensor.
        - ``"camera"`` — nested list from the stacked-frame array.

        Returns:
            dict[str, Any]: Observation dictionary.
        """
        observation = {}
        if self.distance_sensors is not None:
            observation["distance_sensors"] = [
                s.getValue() for s in self.distance_sensors
            ]
        if self.camera is not None:
            observation["camera"] = self.camera.process_camera_image().tolist()
        return observation
