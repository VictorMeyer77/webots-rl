import sys

sys.path.append("../../libraries")

import logging

from brain.controller.epuck.genetic.epuck_turner_genetic_train import EpuckTurnerGeneticTrain
from brain.utils.logger import logger
from controller import Robot

logger.add_console_logger(logging.INFO)
logger.add_file_logger(logging.INFO)


TIME_STEP = 64
MAX_SPEED = 6.28

robot = Robot()

epuck = EpuckTurnerGeneticTrain(robot, TIME_STEP, MAX_SPEED)
epuck.run()
