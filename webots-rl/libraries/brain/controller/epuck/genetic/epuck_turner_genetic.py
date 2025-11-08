import json
import os

from brain.controller.epuck import EpuckTurner
from brain.train_model import MODEL_PATH
from brain.utils.logger import logger
from controller import Robot


class EpuckTurnerGenetic(EpuckTurner):
    """
    Controller for Genetic Algorithm for EpuckTurner.

    Attributes:
        actions (list[int]): List of discrete actions to perform.
    """

    actions: list[int] = []

    def __init__(self, robot: Robot, timestep: int, max_speed: float, model_name: str):
        """
        Initialize the genetic training controller for the e-puck robot.

        Args:
            robot (Robot): The Webots robot instance.
            timestep (int): Simulation timestep in milliseconds.
            max_speed (float): Maximum speed for the robot's motors.
        """
        super().__init__(robot=robot, timestep=timestep, max_speed=max_speed, train=False)

        with open(os.path.join(MODEL_PATH, model_name + ".json"), "r", encoding="utf-8") as f:
            self.actions = json.load(f)  # todo

    def policy(self, _observation: dict) -> int:
        action = self.actions[self.step_index]
        logger().debug(f"Taking action {action}")
        return action
