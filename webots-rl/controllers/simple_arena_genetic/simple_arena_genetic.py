"""Entry point controller for the Simple Arena genetic experiment.

This script is executed by Webots for either:
  1. Training bridge mode (env var TRAIN=1): Handles the controller side
     of the genetic evaluation handshake via `EpuckTurnerGenetic.train()`.
  2. Deployment mode: Loads a persisted genome (`ModelGenetic`) and
     replays the action sequence with `EpuckTurnerGenetic.run()`.

Environment Variable:
  TRAIN:
    - "1" -> training handshake mode.
    - anything else / unset -> deployment (inference) mode.

Logging:
  Initializes console and file loggers at INFO level.
"""

import sys

sys.path.append("../../libraries")

import logging
import os

from brain.controller.epuck.epuck_turner_genetic import EpuckTurnerGenetic
from brain.model.genetic import ModelGenetic
from brain.utils.logger import logger
from controller import Robot

TIME_STEP = 64
MAX_SPEED = 6.28

if __name__ == "__main__":

    train = True if os.getenv("TRAIN") == "1" else False

    logger.add_console_logger(logging.INFO)
    logger.add_file_logger(logging.INFO)

    robot = Robot()
    epuck = EpuckTurnerGenetic(robot, TIME_STEP, MAX_SPEED)

    if train:
        epuck.train()
    else:
        model = ModelGenetic()
        model.load("simple_arena_genetic_UFv0")
        epuck.set_model(model)
        epuck.run()
