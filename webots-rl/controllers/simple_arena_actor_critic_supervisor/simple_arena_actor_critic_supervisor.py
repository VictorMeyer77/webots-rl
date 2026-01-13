"""
Supervisor Script for Actor-Critic Vision-Based Navigation Training.

Main entry point for training or evaluating an Actor-Critic agent in the simple
arena environment. The supervisor coordinates the training process, manages the
environment state, and controls episode resets.

Modes:
    - Training (TRAIN=1): Runs RL training loop for specified epochs
    - Evaluation (TRAIN=0): Runs single episode with trained model for testing

Architecture:
    - Shared CNN feature extractor + separate actor/critic heads
    - Input: (42, 42, 4) stacked grayscale camera frames
    - Actor Output: 4 action probabilities (forward, left, right, backward)
    - Critic Output: Single state value estimate
    - Optimizer: Adam with learning rate 0.001

Training Process:
    1. Build Actor-Critic model
    2. Initialize environment and trainer
    3. Run training for EPOCHS episodes
    4. Save final model to MODEL_PATH/{MODEL_NAME}.keras

Key Hyperparameters:
    - EPOCHS: 2000 training episodes
    - GAMMA: 0.99 discount factor for TD learning
    - LEARNING_RATE: 0.001 for faster convergence
"""

import sys

sys.path.append("../../libraries")

import logging
import os

from brain.environment.simple_arena import EnvironmentSimpleArena
from brain.trainer.simple_arena.actor_critic import TrainerActorCriticSimpleArena
from brain.utils.logger import logger
from controller import Supervisor
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Model Configuration
MODEL_NAME = "simple_arena_actor_critic"  # Base name for saved model files

# Simulation Parameters
TIME_STEP = 64  # Simulation timestep in milliseconds (15.625 Hz)
EPISODE_SIZE = 2000  # Maximum steps per episode before timeout

# Actor-Critic Hyperparameters
EPOCHS = 2000  # Number of training episodes
GAMMA = 0.99  # Discount factor for future rewards [0, 1]

# Neural Network Parameters
LEARNING_RATE = 0.001  # Adam optimizer learning rate


def build_model() -> Model:

    # Input layer
    inputs = Input(shape=(42, 42, 4))

    # Shared feature extractor
    x = Conv2D(32, (4, 4), strides=(2, 2), activation="relu", name="conv1")(inputs)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation="relu", name="conv2")(x)
    x = Flatten(name="flatten")(x)
    x = Dense(256, activation="relu", name="shared_dense")(x)

    # Actor head (policy)
    actor_output = Dense(4, activation="softmax", name="actor")(x)

    # Critic head (value function)
    critic_output = Dense(1, activation="linear", name="critic")(x)

    # Create model with two outputs
    model = Model(inputs=inputs, outputs=[actor_output, critic_output])
    model.summary()

    return model


if __name__ == "__main__":
    # Setup logging
    logger.add_console_logger(logging.INFO)
    logger.add_file_logger(logging.DEBUG)

    # Determine mode from environment variable
    train = True if os.getenv("TRAIN") == "1" else False

    # Initialize Webots supervisor and environment
    supervisor = Supervisor()
    environment = EnvironmentSimpleArena(supervisor, TIME_STEP, EPISODE_SIZE)

    if train:
        # Training mode: Create trainer and run training loop
        trainer = TrainerActorCriticSimpleArena(
            model_name=MODEL_NAME,
            environment=environment,
            model=build_model(),
            gamma=GAMMA,
            optimizer=Adam(learning_rate=LEARNING_RATE),
        )
        trainer.run(EPOCHS)
        trainer.save_model()
    else:
        # Evaluation mode: Run single episode
        final_state = environment.run()
        logger().info(f"Final state: {final_state.to_json()}")

    # Cleanup
    environment.quit()
