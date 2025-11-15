"""Supervisor entry point for the Simple Arena genetic experiment.

Purpose:
  Orchestrates either genetic algorithm training (evolving an action
  sequence genome) or plain environment execution (deployment) for the
  Simple Arena using Webots.

Modes (env var TRAIN):
  TRAIN=1  -> run evolutionary training using `TrainerSimpleArenaGenetic`.
  otherwise -> load and replay the best genome via the controller-side
  environment loop.

Constants:
  MODEL_NAME: Base file name stem used when persisting the best genome.
  TIME_STEP: Webots basic time step (ms) forwarded to Supervisor.step().
  EPISODE_SIZE: Number of discrete actions per genome / episode length.
  MUTATION_RATE: Per-gene mutation probability.
  GENERATION_SIZE: Number of individuals per generation.
  SELECTION_RATE: Fraction of top individuals chosen for breeding.
  EPOCHS: Number of evolutionary generations (training loops).

Note:
  This script is intended to be invoked directly by Webots as a controller.
"""

import sys

sys.path.append("../../libraries")

import logging
import os

from brain.environment.simple_arena import EnvironmentSimpleArena
from brain.train_model.simple_arena.genetic import TrainerSimpleArenaGenetic
from brain.utils.logger import logger
from controller import Supervisor

MODEL_NAME = "simple_arena_genetic"
TIME_STEP = 64
EPISODE_SIZE = 3000
MUTATION_RATE = 0.1
GENERATION_SIZE = 50
SELECTION_RATE = 0.2
EPOCHS = 20

if __name__ == "__main__":

    train = True if os.getenv("TRAIN") == "1" else False

    logger.add_console_logger(logging.INFO)
    logger.add_file_logger(logging.INFO)

    supervisor = Supervisor()
    environment = EnvironmentSimpleArena(supervisor, TIME_STEP, EPISODE_SIZE)

    if train:
        trainer = TrainerSimpleArenaGenetic(
            environment=environment,
            model_name=MODEL_NAME,
            generation_size=GENERATION_SIZE,
            individual_size=EPISODE_SIZE,
            mutation_rate=MUTATION_RATE,
            selection_rate=SELECTION_RATE,
        )
        trainer.run(EPOCHS)
        trainer.save_model()
    else:
        environment.run()

    environment.quit()
