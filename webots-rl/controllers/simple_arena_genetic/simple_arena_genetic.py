import sys

sys.path.append("../../libraries")

import logging
import os

from brain.controller.epuck.genetic.epuck_turner_genetic import EpuckTurnerGenetic
from brain.controller.epuck.genetic.epuck_turner_genetic_train import EpuckTurnerGeneticTrain
from brain.utils.logger import logger
from controller import Robot

"""
Main script to run or train the genetic algorithm-based e-puck controller in the simple arena environment.

This script sets up logging, determines the mode (training or inference) based on the TRAIN environment variable,
and initializes the appropriate controller for the e-puck robot in Webots.

Environment Variables:
    TRAIN: Set to "1" to enable training mode; otherwise, runs in inference mode.

Usage:
    - Set the TRAIN environment variable to "1" to train the controller.
    - Run the script to start the simulation with the selected mode.

Attributes:
    TIME_STEP (int): Simulation time step in milliseconds.
    MAX_SPEED (float): Maximum speed for the e-puck robot.
"""


TIME_STEP = 64
MAX_SPEED = 6.28

train = True if os.getenv("TRAIN") == "1" else False

logger.add_console_logger(logging.INFO)
logger.add_file_logger(logging.INFO)

robot = Robot()

if train:
    epuck = EpuckTurnerGeneticTrain(robot, TIME_STEP, MAX_SPEED)
    epuck.run()
else:
    epuck = EpuckTurnerGenetic(robot, TIME_STEP, MAX_SPEED, "simple_arena_genetic_x7mE")
    epuck.run()
