import logging
import random
from typing import Any

from controller import Supervisor

from corl.environment import Environment
from corl.schemas.learning import Environment as EnvironmentSchema
from corl.trainer.environment import TrainerEnvironment
from corl.utils.config import Config
from corl.utils.logger import setup_logging

logger = logging.getLogger(__name__)


TIME_STEP = 32  # Simulation timestep in milliseconds (15.625 Hz)
MAX_TIMESTEP = 1875  # Maximum episode duration: 1875 * 32 ms = 60 s
ACTION_REPEAT = 25  # Number of simulation timesteps to repeat each action
CENTER_X = 0.0  # X coordinate of the pit centre
CENTER_Z = 0.0  # Z coordinate of the pit centre
ROBOT_DEF = "ROBOT_BB-8"  # Definition name of the BB-8 robot in the Webots scene
PIT_DEF = "PIT"  # Definition name of the pit in the Webots scene
RANDOMIZE = True  # Whether to randomize robot position and yaw between episodes
WARMUP_STEPS = 40  # Simulation timesteps to run before RL loop starts


class EnvironmentPitEscape(Environment):
    """
    Pit-escape environment for the BB-8 spherical robot.

    The robot starts inside a pit and must learn to escape. Each episode
    the robot is dropped from a height with a random orientation. The reward
    is shaped by progress toward the pit edge, with a bonus on escape and a
    penalty on timeout.

    Attributes:
        robot: Webots node handle for the BB-8 robot.
        robot_translation: ``translation`` field of the robot node.
        robot_rotation: ``rotation`` field of the robot node.
        pit_radius: Radius of the pit, read from the scene at init.
        longest_distance: Farthest distance from the pit centre reached
            in the current episode (used for progress shaping).
    """

    robot: Any
    robot_translation: Any
    robot_rotation: Any
    pit_radius: float
    longest_distance: float

    def __init__(
        self,
        supervisor: Supervisor,
        timestep: int,
        max_timestep: int,
    ):
        """
        Initialise the environment, retrieving robot and pit nodes.

        Reads ``pitRadius`` directly from the ``PIT`` node field so the
        environment adapts automatically to different world configurations.

        Args:
            supervisor: Webots Supervisor instance.
            timestep: Simulation timestep in milliseconds.
            max_timestep: Maximum number of simulation steps per episode
                before forced termination.
        """
        super().__init__(
            supervisor=supervisor,
            timestep=timestep,
            max_timestep=max_timestep,
        )
        self.robot = self.supervisor.getFromDef(ROBOT_DEF)
        self.robot_translation = self.robot.getField("translation")
        self.robot_rotation = self.robot.getField("rotation")
        pit = self.supervisor.getFromDef(PIT_DEF)
        self.pit_radius = pit.getField("pitRadius").getSFFloat()
        self.longest_distance = 0.0

    def distance_from_center(self) -> float:
        """
        Compute the Euclidean distance of the robot from the pit centre.

        Uses the X and Z axes (horizontal plane) to match the C supervisor
        logic, since Webots uses a Y-up coordinate system.

        Returns:
            float: Distance from (CENTER_X, CENTER_Z).
        """
        position = self.robot_translation.getSFVec3f()
        dx = position[0] - CENTER_X
        dz = position[2] - CENTER_Z
        return (dx**2 + dz**2) ** 0.5

    def is_success(self) -> bool:
        """
        Return True if the robot has escaped the pit.

        The robot is considered to have escaped once its farthest recorded
        distance from the centre has exceeded the pit radius.

        Returns:
            bool: True if the robot has escaped, False otherwise.
        """
        return self.longest_distance >= self.pit_radius

    def evaluate_training_step(self) -> EnvironmentSchema:
        """
        Compute the reward and termination flag for the current simulation step.

        Reward structure:
            - Flat time penalty every step (``-0.1``) to encourage speed.
            - Flat escape bonus on success: ``+50.0``.
            - ``-5.0`` penalty on timeout without escaping.
            - ``+10 * progress`` shaping reward proportional to the increase
              in farthest distance from the pit centre.

        Returns:
            EnvironmentSchema: Step result containing ``done``, ``reward``,
            and ``data`` keys ``distance_from_center`` and ``is_success``.
        """
        distance = self.distance_from_center()

        progress = max(0.0, distance - self.longest_distance)
        self.longest_distance = max(self.longest_distance, distance)

        is_success = self.is_success()
        is_terminated = self.is_terminated()

        reward = -0.1  # Time penalty

        if is_success:
            reward += 50.0  # Escape bonus
        elif is_terminated:
            reward -= 5.0  # Timeout penalty
        else:
            reward += 10 * progress  # Progress reward

        return EnvironmentSchema(
            done=is_terminated,
            reward=reward,
            data={
                "distance_from_center": distance,
                "is_success": is_success,
            },
        )

    def reset(self) -> None:
        """
        Reset episode bookkeeping and restart robot controller.

        Actions:
            * Restart BB-8 controller.
            * Clear longest-distance record.
            * Delegate base reset (calls ``supervisor.simulationReset()`` and
              clears the step index).
        """
        self.robot.restartController()
        self.longest_distance = 0.0
        super().reset()

    def randomize(self) -> None:
        """
        Reset the robot's vertical position and randomise its yaw.

        Keeps the current X and Y translation unchanged and sets Z to ``0.7``
        so the robot drops and lands naturally. Applies a random yaw (rotation
        around the vertical Y axis) so the robot's initial heading differs
        each episode. Calls ``Node.resetPhysics()`` to clear residual velocity
        so the physics engine accepts the new position. Pitch and roll are not
        randomised — the spherical body self-rights immediately after landing.
        """
        position = self.robot_translation.getSFVec3f()
        self.robot_translation.setSFVec3f([position[0], position[1], 0.7])

        angle = random.uniform(0.0, 2.0 * 3.141592653589793)
        self.robot_rotation.setSFRotation([0, 1, 0, angle])
        logger.info(
            f"Randomized environment: position=({position[0]:.3f}, {position[1]:.3f}, 0.7), yaw={angle:.3f} rad."
        )


if __name__ == "__main__":
    config = Config()
    setup_logging(config)

    supervisor = Supervisor()

    environment = EnvironmentPitEscape(
        supervisor=supervisor,
        timestep=TIME_STEP,
        max_timestep=MAX_TIMESTEP,
    )

    if config.get("train_id") is not None:
        TrainerEnvironment(environment, config, randomize=RANDOMIZE).run(
            action_repeat=ACTION_REPEAT, warmup_steps=WARMUP_STEPS
        )
    else:
        environment.run()
