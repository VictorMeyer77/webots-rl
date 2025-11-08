import sys

sys.path.append("../../libraries")

import logging

from brain.environment.simple_arena.genetic import EnvironmentSimpleArenaGenetic
from brain.model.simple_arena.genetic import SimpleArenaGenetic
from brain.utils.logger import logger
from controller import Supervisor

logger.add_console_logger(logging.INFO)
logger.add_file_logger(logging.INFO)

MAX_STEP = 3000
MUTATION_RATE = 0.1
POPULATION_SIZE = 50
SELECTION_RATE = 0.2
EPOCH = 10

supervisor = Supervisor()
environment = EnvironmentSimpleArenaGenetic(supervisor, MAX_STEP, True)
genetic = SimpleArenaGenetic(environment, POPULATION_SIZE, MAX_STEP, MUTATION_RATE, SELECTION_RATE, EPOCH)

genetic.run()
genetic.save()
