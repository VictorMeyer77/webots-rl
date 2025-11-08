import sys

sys.path.append("../../libraries")

import logging

from brain.controller.epuck.genetic.epuck_turner_genetic import EpuckTurnerGenetic
from brain.utils.logger import logger
from controller import Robot

TIME_STEP = 64
MAX_SPEED = 6.28

logger.add_console_logger(logging.DEBUG)
logger.add_file_logger(logging.DEBUG)

robot = Robot()

epuck = EpuckTurnerGenetic(robot, TIME_STEP, MAX_SPEED)
epuck.run()
