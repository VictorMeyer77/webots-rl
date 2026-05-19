import logging
from typing import Any

from controller import Supervisor

from corl.environment import Environment
from corl.schemas.learning import Environment as EnvironmentSchema
from corl.trainer.environment import TrainerEnvironment
from corl.utils.config import Config
from corl.utils.logger import setup_logging

logger = logging.getLogger(__name__)


TIME_STEP = 32  # Simulation timestep in milliseconds (15.625 Hz)
MAX_TIMESTEP = 1875  # Maximum episode duration in seconds (simulation time)
ACTION_REPEAT = 25  # Number of simulation timesteps to repeat each action
EPUCK_DEF = "EPUCK"  # Definiton name of the EPUCK robot in the Webots scene
FINISH_LINE_DEF = (
    "FINISH_LINE"  # Definition name of the finish line in the Webots scene
)
FINISH_DISTANCE_THRESHOLD = (
    0.03  # Threshold distance for successful completion of the episode
)


class EnvironmentSimpleArena(Environment):
    epuck: Any
    epuck_translation: Any
    finish_line_translation: Any
    last_finish_distance: float | None

    def __init__(
        self,
        supervisor: Supervisor,
        timestep: int,
        max_timestep: int,
    ):
        """
        Initialize the environment, retrieving agent and finish line nodes.

        Args:
            supervisor (Supervisor): Webots Supervisor instance.
            timestep (int): Simulation timestep.
            max_timestep (int): Maximum number of simulation steps per episode before forced termination.
        """
        super().__init__(
            supervisor=supervisor,
            timestep=timestep,
            max_timestep=max_timestep,
        )
        self.epuck = self.supervisor.getFromDef(EPUCK_DEF)
        self.epuck_translation = self.epuck.getField("translation")
        finish_line = self.supervisor.getFromDef(FINISH_LINE_DEF)
        self.finish_line_translation = finish_line.getField("translation")
        self.last_finish_distance = None

    def finish_distance(self) -> float:
        """
        Compute the Euclidean distance between agent and finish line.

        Returns:
            float: Distance to the finish line.
        """
        epuck_position = self.epuck_translation.getSFVec3f()
        finish_line_position = self.finish_line_translation.getSFVec3f()
        dx = finish_line_position[0] - epuck_position[0]
        dy = finish_line_position[1] - epuck_position[1]
        distance = (dx**2 + dy**2) ** 0.5
        return distance

    def is_success(self) -> bool:
        """
        Return True if the agent is within the finish distance threshold.

        Returns:
            bool: True if the agent has reached the finish line, False otherwise.
        """
        return self.finish_distance() < FINISH_DISTANCE_THRESHOLD

    def evaluate_training_step(self) -> EnvironmentSchema:
        """
        Compute the reward and termination flag for the current simulation step.

        Reward structure:
            - ``-0.01`` time penalty every step to encourage speed.
            - ``+50.0`` bonus on success (reaching the finish line).
            - ``-5.0`` penalty on timeout without success.
            - ``+10 * progress`` shaping reward proportional to distance closed
              toward the finish line since the last step.

        Returns:
            EnvironmentSchema: Step result containing ``done``, ``reward``, and
            ``data`` keys ``finish_line_distance`` and ``is_success``.
        """
        finish_line_distance = self.finish_distance()
        is_success = self.is_success()
        is_terminated = self.is_terminated()

        reward = -0.01  # Time penalty

        if is_success:
            reward += 50.0  # Success bonus
        elif is_terminated:
            reward -= 5.0  # Failure penalty
        elif self.last_finish_distance is not None:
            progress = self.last_finish_distance - finish_line_distance
            reward += 10 * progress  # Progress reward

        self.last_finish_distance = finish_line_distance

        return EnvironmentSchema(
            done=is_terminated,
            reward=reward,
            data={
                "finish_line_distance": finish_line_distance,
                "is_success": is_success,
            },
        )

    def randomize(self) -> None:
        """Not used for this environment."""

    def reset(self) -> None:
        """
        Reset episode bookkeeping and restart robot controller.

        Actions:
            * Restart e-puck controller.
            * Clear distance history.
            * Delegate base reset (clears step index and sets initial state).
        """
        self.epuck.restartController()
        self.last_finish_distance = None
        super().reset()


if __name__ == "__main__":
    config = Config()
    setup_logging(config)

    supervisor = Supervisor()

    environment = EnvironmentSimpleArena(
        supervisor=supervisor,
        timestep=TIME_STEP,
        max_timestep=MAX_TIMESTEP,
    )

    if config.get("train_id") is not None:
        TrainerEnvironment(environment, config).run(ACTION_REPEAT)
    else:
        environment.run()
