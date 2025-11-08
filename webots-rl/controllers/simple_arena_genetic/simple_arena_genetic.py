import sys

sys.path.append("../../libraries")

import logging

from brain.controller.epuck.genetic.epuck_turner_genetic import EpuckTurnerGenetic
from brain.controller.epuck.genetic.epuck_turner_genetic_train import EpuckTurnerGeneticTrain
from brain.utils.logger import logger
from controller import Robot
import os

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
	epuck = EpuckTurnerGenetic(robot, TIME_STEP, MAX_SPEED, "simple_arena_genetic_GaxM")
	epuck.run()



