"""
Main controller script for e-puck robot using REINFORCE policy gradient learning.

This script serves as the entry point for the simple_arena_reinforce controller in Webots.
It supports both training mode (with supervisor communication) and inference mode (using
pre-trained policy models).

Training Mode:
    - Enabled by setting environment variable TRAIN=1
    - Uses distance sensors for state observations
    - Communicates with supervisor via emitter/receiver for REINFORCE updates
    - Supervisor handles policy gradient computation and parameter updates

Inference Mode:
    - Runs when TRAIN is not set or TRAIN=0
    - Loads pre-trained policy network from disk
    - Executes autonomous navigation using learned policy
    - No communication with supervisor required

Constants:
    TIME_STEP (int): Simulation timestep in milliseconds (64ms ≈ 15.6 Hz).
        Defines controller update frequency and sensor sampling rate.
    MAX_SPEED (float): Maximum wheel motor angular velocity in rad/s (6.28 ≈ 1 rev/s).
        Upper limit for differential drive actuation.

Environment Variables:
    TRAIN: Controls execution mode.
        - "1": Enable training mode with supervisor synchronization
        - Other/unset: Enable inference mode with pre-trained model

Model Loading:
    - Inference mode loads model "simple_arena_reinforce_8zAT.keras"
    - Model file must exist in brain.model.MODEL_PATH directory
    - Change model name in epuck.load_model() call to use different policies
"""

import sys

sys.path.append("../../libraries")

import logging
import os

from brain.controller.epuck.epuck_turner_reinforce import EpuckTurnerReinforce
from brain.utils.logger import logger
from controller import Robot

TIME_STEP = 64
MAX_SPEED = 6.28


if __name__ == "__main__":

    logger.add_console_logger(logging.INFO)
    logger.add_file_logger(logging.INFO)

    train = True if os.getenv("TRAIN") == "1" else False
    robot = Robot()
    epuck = EpuckTurnerReinforce(robot, TIME_STEP, MAX_SPEED)
    epuck.init_distance_sensors()

    if train:
        epuck.init_emitter_receiver()
        epuck.train()
    else:
        epuck.load_model("simple_arena_reinforce_8zAT")
        epuck.run()
