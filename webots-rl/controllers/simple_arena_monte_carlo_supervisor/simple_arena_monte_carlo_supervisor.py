"""
Supervisor launcher for Monte Carlo training or evaluation in the simple arena.

Purpose:
    Create the Webots Supervisor, environment, and (optionally) a Monte Carlo trainer.
    Run training epochs or a single evaluation episode based on the TRAIN environment variable.

Environment Variable:
    TRAIN=1 -> training mode
    TRAIN unset or not equal to 1 -> evaluation mode

Constants:
    TIME_STEP: Webots simulation step (ms).
    EPISODE_SIZE: Maximum steps per episode.
    EPOCHS: Training epochs for Monte Carlo (episodes).
    GAMMA: Discount factor.
    EPSILON: Initial exploration rate.

Logging:
    Console and file loggers are configured before execution.
"""

import sys

sys.path.append("../../libraries")

import logging
import os

from brain.environment.simple_arena import EnvironmentSimpleArena
from brain.trainer.simple_arena.monte_carlo import TrainerMonteCarloSimpleArena
from brain.utils.logger import logger
from controller import Supervisor

MODEL_NAME = "simple_arena_monte_carlo"
TIME_STEP = 64
EPISODE_SIZE = 2000
EPOCHS = 2000
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.998

if __name__ == "__main__":
    logger.add_console_logger(logging.INFO)
    logger.add_file_logger(logging.INFO)

    train = True if os.getenv("TRAIN") == "1" else False
    supervisor = Supervisor()
    environment = EnvironmentSimpleArena(supervisor, TIME_STEP, EPISODE_SIZE)

    if train:
        trainer = TrainerMonteCarloSimpleArena(
            model_name=MODEL_NAME,
            environment=environment,
            gamma=GAMMA,
            epsilon=EPSILON,
            epsilon_decay=EPSILON_DECAY,
        )
        trainer.run(EPOCHS)
        trainer.save_model()
    else:
        final_state = environment.run()
        logger().info(f"Final state: {final_state.to_json()}")

    environment.quit()
