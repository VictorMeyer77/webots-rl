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
