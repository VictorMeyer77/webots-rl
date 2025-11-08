import json

from brain.controller.epuck import EpuckTurner
from brain.utils.logger import logger
from controller import Robot


class EpuckTurnerGeneticTrain(EpuckTurner):
    """
    Controller for training Genetic Algorithm for EpuckTurner.

    Attributes:
        actions (list[int]): List of discrete actions to perform.
    """

    actions: list[int] = []

    def __init__(self, robot: Robot, timestep: int, max_speed: float):
        """
        Initialize the genetic training controller for the e-puck robot.

        Args:
            robot (Robot): The Webots robot instance.
            timestep (int): Simulation timestep in milliseconds.
            max_speed (float): Maximum speed for the robot's motors.
        """
        super().__init__(robot=robot, timestep=timestep, max_speed=max_speed, train=True)

    def policy(self, _observation: dict) -> int:
        """
        Determine the next action for the robot based on received TCP messages from Supervisor.

        Reads a message from the TCP socket. If a new message is received, updates the actions list
        and resets the robot. Returns the action corresponding to the current step index.

        Args:
            _observation (dict): Current observation (unused).

        Returns:
            int: The action to execute at the current step.
        """
        message = self.tcp_socket.read()
        if message != "":
            self.actions = json.loads(message)["actions"]
            self.reset()

        action = self.actions[self.step_index]
        logger().debug(f"Step {self.step_index}: taking action {action}")
        return action
