"""
E-puck Controller for Deep Q-Learning Vision-Based Navigation.

Main controller script for running the e-puck robot with a trained Deep Q-Learning
CNN model. Supports two modes:
    - Training mode (TRAIN=1): Communicates with supervisor for RL training
    - Inference mode (TRAIN=0): Autonomous navigation using pre-trained model

Usage:
    Training: export TRAIN=1 && webots world.wbt
    Inference: export TRAIN=0 && webots world.wbt (or just run without TRAIN set)

The robot uses camera input only, processed through a CNN to select actions.
"""

import sys

sys.path.append("../../libraries")

import logging
import os

from brain.controller.epuck.epuck_turner_deep_q_table import EpuckTurnerDeepQTable
from brain.utils.logger import logger
from controller import Robot

TIME_STEP = 64
MAX_SPEED = 6.28


if __name__ == "__main__":

    logger.add_console_logger(logging.INFO)
    logger.add_file_logger(logging.INFO)

    train = True if os.getenv("TRAIN") == "1" else False
    robot = Robot()
    epuck = EpuckTurnerDeepQTable(robot, TIME_STEP, MAX_SPEED)
    epuck.init_camera()

    if train:
        epuck.init_emitter_receiver()
        epuck.train()
    else:
        epuck.load_model("simple_arena_deep_q_learning_UEhY")
        epuck.run()
