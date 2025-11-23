"""
Supervisor launcher for tabular Double Q-learning training or evaluation in the simple arena.

Overview:
    This script instantiates the Webots Supervisor, wraps it with an
    ``EnvironmentSimpleArena``, and optionally runs multiple Double
    Q-learning training episodes (epochs). In evaluation mode it runs a
    single episode without learning and logs the final state.

Execution Modes (env var ``TRAIN``):
    TRAIN=1   -> Training mode (runs ``EPOCHS`` Double Q-learning episodes, saves model).
    Otherwise -> Evaluation mode (single episode, no updates).

Constants:
    MODEL_NAME: Base name for the persisted Q-table (``<MODEL_NAME>.npy``).
    TIME_STEP: Supervisor loop timestep (ms).
    EPISODE_SIZE: Maximum number of steps per episode before forced termination.
    EPOCHS: Number of training episodes in training mode.
    ALPHA: Double Q-learning learning rate.
    GAMMA: Discount factor for future rewards.
    EPSILON: Initial exploration rate (decayed inside the trainer).
"""

import sys

sys.path.append("../../libraries")

import logging
import os

from brain.environment.simple_arena import EnvironmentSimpleArena
from brain.trainer.simple_arena.double_q_learning import TrainerDoubleQLearningSimpleArena
from brain.utils.logger import logger
from controller import Supervisor

MODEL_NAME = "simple_arena_double_q_learning"
TIME_STEP = 64
EPISODE_SIZE = 3000
EPOCHS = 2500
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0

if __name__ == "__main__":
    logger.add_console_logger(logging.INFO)
    logger.add_file_logger(logging.INFO)

    train = True if os.getenv("TRAIN") == "1" else False
    supervisor = Supervisor()
    environment = EnvironmentSimpleArena(supervisor, TIME_STEP, EPISODE_SIZE)

    if train:
        trainer = TrainerDoubleQLearningSimpleArena(
            model_name=MODEL_NAME,
            environment=environment,
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon=EPSILON,
        )
        trainer.run(EPOCHS)
        trainer.save_model()
    else:
        final_state = environment.run()
        logger().info(f"Final state: {final_state.to_json()}")

    environment.quit()
