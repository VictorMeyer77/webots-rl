from typing import Any

import numpy as np
from controller import Robot

from corl.agent.epuck.discrete import EpuckDiscrete
from corl.model.model import Model
from corl.model.value_table import ModelValueTable
from corl.trainer.agent import TrainerAgent
from corl.utils.config import Config
from corl.utils.logger import setup_logging

TIME_STEP = 32  # Simulation timestep in milliseconds (15.625 Hz)
ACTION_REPEAT = 25  # Number of simulation timesteps to repeat each action
MAX_TIMESTEP = (
    1875  # Maximum number of simulation timesteps before reset (1875 * 32 ms = 60s)
)


class EpuckDoubleQLearningController(EpuckDiscrete):
    """
    E-puck controller driven by a Double Q-learning value-table policy.

    Discretises distance-sensor readings into bins and looks up the
    greedy action from a pre-trained Double Q-table.
    """

    def __init__(
        self,
        robot: Robot,
        timestep: int,
        action_repeat: int,
        model: Model | None = None,
    ):
        """
        Initialise the controller and activate distance sensors.

        Args:
            robot (Robot): Webots Robot instance.
            timestep (int): Simulation timestep in milliseconds.
            action_repeat (int): Number of simulation steps each chosen action
                is held for.
            model (Model | None): Pre-trained value-table model. Pass ``None``
                during training; the trainer will supply actions externally.
        """
        super().__init__(
            robot=robot,
            timestep=timestep,
            action_repeat=action_repeat,
            model=model,
        )
        self.init_distance_sensors()

    def policy(self, observation: dict[str, Any]) -> list[float]:
        """
        Return the greedy action for the current observation.

        Distance sensor readings are discretised into three bins
        (``[70, 80]``) before being passed to the value table.

        Args:
            observation (dict[str, Any]): Must contain a ``"distance_sensors"``
                key with a list of raw sensor readings.

        Returns:
            list[float]: Single-element list with the greedy action index as a
                float (e.g. ``[3.0]``).

        Raises:
            ValueError: If ``"distance_sensors"`` is absent from ``observation``.
        """
        if "distance_sensors" in observation:
            binned_observation = np.digitize(
                np.asarray(observation["distance_sensors"], dtype=np.float32),
                bins=[70, 80],
            )
            return [float(self.model.predict(binned_observation.astype(np.float32)))]
        else:
            raise ValueError("Observation missing 'distance_sensors' key.")


if __name__ == "__main__":
    config = Config()
    setup_logging(config)

    robot = Robot()

    if config.get("train_id") is not None:
        epuck = EpuckDoubleQLearningController(
            robot=robot,
            timestep=TIME_STEP,
            action_repeat=ACTION_REPEAT,
            model=None,
        )
        TrainerAgent(epuck, config).run(MAX_TIMESTEP)
    else:
        model = ModelValueTable(
            model_dir="/Users/victormeyer/Dev/Self/webots-rl/projects/.model/double_q_learning"
        )

        epuck = EpuckDoubleQLearningController(
            robot=robot,
            timestep=TIME_STEP,
            action_repeat=ACTION_REPEAT,
            model=model,
        )
        epuck.run()
