"""
Controller runner for the simple arena tabular SARSA setup.

Purpose:
    Launch an e-puck controller (`EpuckTurnerQTable`) inside Webots in either:
      1. Training mode (`TRAIN=1`): controller enters its message-driven `train()` loop
         cooperating with the supervisor (which performs learning).
      2. Evaluation mode: loads a persisted SARSA `ModelQTable` and runs `run()` with a greedy policy.

Environment Variable:
    TRAIN=1 -> training mode.
    Any other value or unset -> evaluation mode.

Constants:
    TIME_STEP (int): Supervisor/controller simulation timestep in milliseconds.
    MAX_SPEED (float): Maximum wheel motor angular velocity (rad/s).
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
        model.load("simple_arena_sarsa_f4KP")
        epuck.set_model(model)
        epuck.run()
