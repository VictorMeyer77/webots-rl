import sys

sys.path.append("../../libraries")

import logging
import os

from brain.environment.simple_arena import EnvironmentSimpleArena
from brain.train_model.simple_arena.genetic import TrainSimpleArenaGenetic
from brain.utils.logger import logger
from controller import Supervisor

"""
Main script to run or train a genetic algorithm-based supervisor
for the e-puck robot in the simple arena environment using Webots.

This script configures logging, determines the mode (training or inference) based on the TRAIN environment variable,
and initializes the appropriate supervisor and environment for the simulation.

Environment Variables:
    TRAIN: Set to "1" to enable training mode; otherwise, runs in inference mode.

Usage:
    - Set the TRAIN environment variable to "1" to train the supervisor.
    - Run the script to start the simulation in the selected mode.

Attributes:
    MAX_STEP (int): Maximum number of simulation steps per episode.

"""

MAX_STEP = 3000

logger.add_console_logger(logging.INFO)
logger.add_file_logger(logging.INFO)


train = True if os.getenv("TRAIN") == "1" else False

supervisor = Supervisor()

environment = EnvironmentSimpleArena(supervisor, MAX_STEP)

if train:
    MUTATION_RATE = 0.1
    POPULATION_SIZE = 50
    SELECTION_RATE = 0.2
    EPOCH = 10
    model = TrainSimpleArenaGenetic(environment, POPULATION_SIZE, MAX_STEP, MUTATION_RATE, SELECTION_RATE, EPOCH)
    model.run()
    environment.quit()
    model.save()
else:
    environment.run()
    environment.quit()
