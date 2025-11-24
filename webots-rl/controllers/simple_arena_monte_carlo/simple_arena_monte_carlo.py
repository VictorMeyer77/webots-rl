"""
Module: simple arena Monte Carlo controller runner.

Purpose:
    Launch an e-puck controller (`EpuckTurnerQTable`) inside Webots either in:
        1. Training mode (TRAIN=1) where the controller runs its internal `train()` loop.
        2. Evaluation mode where a saved Monte Carlo model is loaded and bound to the controller.

Environment Variable:
    TRAIN=1 enables training. Any other value or unset executes evaluation.

Constants:
    TIME_STEP (int): Webots simulation step in milliseconds.
    MAX_SPEED (float): Maximum wheel motor velocity (rad/s).
    EPISODE_SIZE (int): Maximum episode steps (used by training logic if referenced).

Model Loading (evaluation):
    - Expects a file named like `simple_arena_monte_carlo_005f` resolvable by `ModelQTable.load()`.
      Adjust the name as needed for your saved checkpoints.

Logging:
    Uses `brain.utils.logger.logger` helpers to add console and file handlers at INFO level.
"""

import sys

sys.path.append("../../libraries")

import logging
import os

from brain.controller.epuck.epuck_turner_q_table import EpuckTurnerQTable
from brain.model.q_table import ModelQTable
from brain.utils.logger import logger
from controller import Robot

TIME_STEP = 64
MAX_SPEED = 6.28


if __name__ == "__main__":

    logger.add_console_logger(logging.INFO)
    logger.add_file_logger(logging.INFO)

    train = True if os.getenv("TRAIN") == "1" else False
    robot = Robot()
    epuck = EpuckTurnerQTable(robot, TIME_STEP, MAX_SPEED)

    if train:
        epuck.train()
    else:
        model = ModelQTable(observation_cardinality=3)
        model.load("simple_arena_monte_carlo_NoRL")
        epuck.set_model(model)
        epuck.run()
