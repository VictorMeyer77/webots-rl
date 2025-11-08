import sys

sys.path.append("../../libraries")

import logging

from brain.environment.simple_arena.genetic import EnvironmentSimpleArenaGenetic
from brain.model.simple_arena.genetic import SimpleArenaGenetic
from brain.utils.logger import logger
from controller import Supervisor
import os

MAX_STEP = 3000

logger.add_console_logger(logging.DEBUG)
logger.add_file_logger(logging.INFO)


train = True if os.getenv("TRAIN") == "1" else False

supervisor = Supervisor()

environment = EnvironmentSimpleArenaGenetic(supervisor, MAX_STEP, train)

if train:
	MUTATION_RATE = 0.1
	POPULATION_SIZE = 1
	SELECTION_RATE = 0.2
	EPOCH = 1
	model = SimpleArenaGenetic(environment, POPULATION_SIZE, MAX_STEP, MUTATION_RATE, SELECTION_RATE, EPOCH)
	model.run()
	environment.quit()
	model.save()
else:
	environment.run()
	environment.quit()
