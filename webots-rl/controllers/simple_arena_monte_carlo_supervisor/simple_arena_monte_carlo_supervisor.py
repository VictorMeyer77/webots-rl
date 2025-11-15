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
    ACTION_SIZE: Number of discrete actions expected by the trainer.
    OBSERVATION_SIZE: Number of distance sensor readings.
    OBSERVATION_CARDINALITY: Discrete bins per sensor.
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
from brain.train_model.simple_arena.monte_carlo import TrainerMonteCarloSimpleArena
from brain.utils.logger import logger
from controller import Supervisor

MODEL_NAME = "simple_arena_monte_carlo"
TIME_STEP = 64
EPISODE_SIZE = 3000
ACTION_SIZE = 3
OBSERVATION_SIZE = 8
OBSERVATION_CARDINALITY = 3
EPOCHS = 1000
GAMMA = 0.99
EPSILON = 1.0

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
            action_size=ACTION_SIZE,
            observation_size=OBSERVATION_SIZE,
            observation_cardinality=OBSERVATION_CARDINALITY,
            gamma=GAMMA,
            epsilon=EPSILON,
        )
        trainer.run(EPOCHS)
        trainer.save_model()
    else:
        final_state = environment.run()
        logger().info(f"Final state: {final_state.to_json()}")

    environment.quit()
