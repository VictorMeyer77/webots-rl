"""
Simple Arena A2C controller entry point for E-puck robot.

This module serves as the main entry point for the E-puck robot controller in
the Simple Arena environment using the A2C (Advantage Actor-Critic) algorithm.
It initializes the robot hardware interface, configures the actor controller,
and executes either training or inference mode based on environment variables.

The controller runs within the Webots robot process and communicates with the
supervisor for training coordination. In training mode, it exchanges messages
with the supervisor to synchronize observations and actions. In inference mode,
it loads a pre-trained model and runs autonomously.

Constants:
    TIME_STEP: Webots simulation timestep in milliseconds (64ms). Determines
        the frequency of control updates and sensor readings.
    MAX_SPEED: Maximum angular velocity for E-puck motors in radians per second
        (6.28 rad/s ≈ 1 revolution per second). Used to normalize motor commands.

Environment Variables:
    TRAIN: Controls execution mode. Set to "1" for training mode, any other
        value or unset for inference mode.

Usage:
    Training mode:
        TRAIN=1 webots simple_arena_a2c.wbt

    Inference mode:
        webots simple_arena_a2c.wbt
"""

import sys

sys.path.append("../../libraries")

import logging
import os

from brain.controller.epuck.epuck_turner_actor import EpuckTurnerActor
from brain.utils.logger import logger
from controller import Robot

TIME_STEP = 64
MAX_SPEED = 6.28


if __name__ == "__main__":

    logger.add_console_logger(logging.INFO)
    logger.add_file_logger(logging.INFO)

    train = True if os.getenv("TRAIN") == "1" else False
    robot = Robot()
    epuck = EpuckTurnerActor(robot, TIME_STEP, MAX_SPEED, stochastic=True)
    epuck.init_camera()

    if train:
        epuck.init_emitter_receiver()
        epuck.train()
    else:
        epuck.load_model("simple_arena_a2c_OiXC")
        epuck.run()
