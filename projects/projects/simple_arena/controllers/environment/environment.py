from corl.trainer.environment import TrainerEnvironment

from corl.utils.config import Config
from corl.utils.logger import setup_logging

import logging
from typing import Any

from controller import Supervisor
from corl.schemas.learning import Environment as EnvironmentSchema
from corl.environment import Environment

logger = logging.getLogger(__name__)



# Simulation Parameters
TIME_STEP = 32  # Simulation timestep in milliseconds (15.625 Hz)
MAX_TIMESTEP = 1875  # Maximum episode duration in seconds (simulation time)
ACTION_REPEAT = 25  # Number of simulation timesteps to repeat each action



EPUCK_DEF = "EPUCK"
FINISH_LINE_DEF = "FINISH_LINE"
FINISH_DISTANCE_THRESHOLD = 0.03

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
            max_step (int): Maximum steps per episode.
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

    def step(self) -> EnvironmentSchema:

        finish_line_distance = self.finish_distance()
        is_success = finish_line_distance < FINISH_DISTANCE_THRESHOLD
        is_terminated = is_success or self.timestep_index >= self.max_timestep

        reward = 0.0

        if is_success:
            reward += 10.0  # Success bonus
        elif is_terminated:
            reward -= 2.0  # Failure penalty
        elif self.last_finish_distance is not None:
            progress = self.last_finish_distance - finish_line_distance
            reward += 0.5 * progress  # Progress reward
            if abs(self.last_finish_distance - finish_line_distance) < 0.001:
                reward -= 0.01  # Stagnation penalty

        self.last_finish_distance = finish_line_distance

        return EnvironmentSchema(
            done=is_terminated,
            reward=reward,
            data={
                "finish_line_distance": finish_line_distance,
                "is_success": is_success,
            },
        )

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
    # Determine mode from environment variable

    config = Config()
    setup_logging(config)

    # Initialize Webots supervisor and environment
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
