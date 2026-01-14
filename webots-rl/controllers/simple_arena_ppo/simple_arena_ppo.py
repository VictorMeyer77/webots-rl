"""
Simple Arena PPO controller entry point for E-puck robot.

This module serves as the main entry point for the E-puck robot controller in
the Simple Arena environment using the PPO (Proximal Policy Optimization)
algorithm. It initializes the robot hardware interface, configures the actor
controller, and executes either training or inference mode based on environment
variables.

The controller runs within the Webots robot process and communicates with the
supervisor for training coordination. In training mode, it exchanges messages
with the supervisor to synchronize observations and actions. In inference mode,
it loads a pre-trained PPO model and runs autonomously.

PPO is an on-policy actor-critic algorithm that improves upon traditional policy
gradient methods by using a clipped surrogate objective to prevent excessively
large policy updates. This makes training more stable and sample-efficient
compared to vanilla policy gradients.

Constants:
    TIME_STEP (int): Webots simulation timestep in milliseconds (64ms).
        Determines the frequency of control updates and sensor readings.
        64ms corresponds to approximately 15.625 Hz update rate.

    MAX_SPEED (float): Maximum angular velocity for E-puck motors in radians
        per second (6.28 rad/s ≈ 1 revolution per second). Used to normalize
        motor commands to the valid range for the E-puck platform.

Environment Variables:
    TRAIN (str): Controls execution mode. Set to "1" for training mode, any
        other value or unset for inference mode.
        - "1": Training mode - communicates with supervisor via emitter/receiver
        - Other/unset: Inference mode - runs pre-trained model autonomously
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
        epuck.load_model("simple_arena_ppo_UTuQ")
        epuck.run()
