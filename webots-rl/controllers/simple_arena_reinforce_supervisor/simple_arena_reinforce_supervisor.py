"""
REINFORCE Policy Gradient Supervisor for Simple Arena Environment.

This script serves as the Webots supervisor controller for training and evaluating
reinforcement learning agents using the REINFORCE (Monte Carlo Policy Gradient) algorithm
in the simple_arena environment.

The supervisor manages:
- Episode orchestration (resets, termination, reward computation)
- Policy gradient training with entropy regularization
- Model persistence and evaluation
- Communication with e-puck robot controller

Training Mode:
    - Enabled by setting environment variable TRAIN=1
    - Collects full episode trajectories using current policy
    - Computes policy gradients with baseline subtraction
    - Updates policy network using batch gradient ascent
    - Saves trained model after completion

Evaluation Mode:
    - Runs when TRAIN is not set or TRAIN=0
    - Executes single episode with current policy
    - Logs final episode statistics
    - No training updates performed

Constants:
    MODEL_NAME (str): Base name for saved model files (without .keras extension).
        Timestamp suffix added during save to prevent overwrites.

    TIME_STEP (int): Simulation timestep in milliseconds (64ms ≈ 15.6 Hz).
        Must match controller timestep for synchronization.

    EPISODE_SIZE (int): Maximum steps per episode before timeout (2000).
        Episode terminates early if goal reached or collision occurs.

    EPOCHS (int): Number of training episodes to execute (2000).
        Total training duration ≈ EPOCHS × avg_episode_length × TIME_STEP.

    GAMMA (float): Discount factor for future rewards (0.99).
        Range: [0, 1]. Higher values prioritize long-term rewards.

    EPISODES_PER_BATCH (int): Number of episodes per gradient update (10).
        Larger batches reduce variance but require more memory.

    ENTROPY_BETA (float): Entropy regularization coefficient (0.01).
        Encourages exploration by penalizing deterministic policies.

    NORMALIZE_RETURNS (bool): Whether to normalize returns using mean/std (True).
        Improves training stability by reducing gradient variance.

    LEARNING_RATE (float): Adam optimizer learning rate (0.0003).
        Controls policy update step size.

Environment Variables:
    TRAIN: Controls execution mode.
        - "1": Enable training mode with policy gradient updates
        - Other/unset: Enable evaluation mode (single episode)

Model Architecture:
    Input: 8 distance sensor values (normalized)
    Hidden: 2 fully-connected layers (64 units each, ReLU activation)
    Output: 3 action logits (no activation, used for softmax policy)

    Total parameters: ~5,000
    Optimizer: Adam with configurable learning rate
"""

import sys

sys.path.append("../../libraries")

import logging
import os

from brain.environment.simple_arena import EnvironmentSimpleArena
from brain.trainer.simple_arena.reinforce import TrainerReinforceSimpleArena
from brain.utils.logger import logger
from controller import Supervisor
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Model Configuration
MODEL_NAME = "simple_arena_reinforce"

TIME_STEP = 64
EPISODE_SIZE = 2000

EPOCHS = 2000
GAMMA = 0.99
EPISODES_PER_BATCH = 10
ENTROPY_BETA = 0.01
NORMALIZE_RETURNS = True

LEARNING_RATE = 0.0003


def build_model() -> Sequential:
    """
    Build a feedforward policy network for REINFORCE.

    Creates a 2-layer fully-connected neural network that outputs action logits
    for the policy distribution. The network maps distance sensor observations
    to action preferences without final activation (logits used for softmax).

    Architecture:
        - Input layer: 8 features (distance sensors)
        - Hidden layer 1: 64 units with ReLU activation
        - Hidden layer 2: 64 units with ReLU activation
        - Output layer: 3 units (action logits, no activation)

    Compilation:
        - Optimizer: Adam with LEARNING_RATE
        - No loss function (custom policy gradient loss applied in trainer)

    Returns:
        Sequential: Compiled Keras model ready for REINFORCE training.
            - Input shape: (batch_size, 8)
            - Output shape: (batch_size, 3)
            - Total params: ~5,000 trainable parameters
    """
    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(8,)))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(3, activation=None))  # logits for 3 actions
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE))
    model.summary()
    return model


if __name__ == "__main__":
    # Setup logging
    logger.add_console_logger(logging.INFO)
    logger.add_file_logger(logging.INFO)

    # Determine mode from environment variable
    train = True if os.getenv("TRAIN") == "1" else False

    # Initialize Webots supervisor and environment
    supervisor = Supervisor()
    environment = EnvironmentSimpleArena(supervisor, TIME_STEP, EPISODE_SIZE)

    if train:
        # Training mode: Create trainer and run training loop
        trainer = TrainerReinforceSimpleArena(
            model_name=MODEL_NAME,
            environment=environment,
            model=build_model(),
            gamma=GAMMA,
            episodes_per_batch=EPISODES_PER_BATCH,
            entropy_beta=ENTROPY_BETA,
            normalize_returns=NORMALIZE_RETURNS,
        )
        trainer.run(EPOCHS)
        trainer.save_model()
    else:
        # Evaluation mode: Run single episode
        final_state = environment.run()
        logger().info(f"Final state: {final_state.to_json()}")

    # Cleanup
    environment.quit()
