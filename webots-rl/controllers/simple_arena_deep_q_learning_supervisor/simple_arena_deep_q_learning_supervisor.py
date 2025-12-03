"""
Supervisor Script for Deep Q-Learning Vision-Based Navigation Training.

Main entry point for training or evaluating a Deep Q-Learning agent in the simple
arena environment. The supervisor coordinates the training process, manages the
environment state, and controls episode resets.

Modes:
    - Training (TRAIN=1): Runs RL training loop for specified epochs
    - Evaluation (TRAIN=0): Runs single episode with trained model for testing

Architecture:
    - CNN with 2 Conv2D layers + Dense layers
    - Input: (42, 42, 4) stacked grayscale camera frames
    - Output: 4 Q-values for actions (forward, left, right, backward)
    - Loss: Huber loss (robust to outliers)
    - Optimizer: Adam with learning rate 0.0001

Training Process:
    1. Build CNN model
    2. Initialize environment and trainer
    3. Run training for EPOCHS episodes
    4. Save final model to MODEL_PATH/{MODEL_NAME}.keras

Key Hyperparameters:
    - EPOCHS: 2000 training episodes
    - MEMORY_SIZE: 200k experiences in replay buffer
    - BATCH_SIZE: 64 experiences per training step
    - GAMMA: 0.99 discount factor
    - EPSILON: 1.0 → decays to ~0.01 for exploration
    - FIT_FREQUENCY: Train every 50 steps
    - UPDATE_TARGET: Update target network every 1000 steps
"""

import sys

from tensorflow.keras.losses import Huber

sys.path.append("../../libraries")

import logging
import os

from brain.environment.simple_arena import EnvironmentSimpleArena
from brain.trainer.simple_arena.deep_q_learning import TrainerDeepQLearningSimpleArena
from brain.utils.logger import logger
from controller import Supervisor
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Model Configuration
MODEL_NAME = "simple_arena_deep_q_learning"  # Base name for saved model files

# Simulation Parameters
TIME_STEP = 64  # Simulation timestep in milliseconds (15.625 Hz)
EPISODE_SIZE = 2000  # Maximum steps per episode before timeout

# Deep Q-Learning Hyperparameters
MEMORY_SIZE = 100000  # Prioritized replay buffer capacity (experiences)
EPOCHS = 2000  # Number of training episodes
GAMMA = 0.99  # Discount factor for future rewards [0, 1]
EPSILON = 1.0  # Initial exploration rate (100% random actions)
EPSILON_DECAY = 0.999  # Multiplicative decay per epoch (reaches ~0.13 after 2000 epochs)
BATCH_SIZE = 64  # Number of experiences sampled per training step
FIT_FREQUENCY = 50  # Train model every N environment steps
UPDATE_TARGET_MODEL_FREQUENCY = 1000  # Update target network every N steps

# Prioritized Experience Replay Parameters
PER_ALPHA = 0.6  # Priority exponent (0=uniform, 1=full prioritization)
PER_BETA_START = 0.4  # Initial importance sampling weight (annealed to 1.0)

# Neural Network Parameters
LEARNING_RATE = 0.0001  # Adam optimizer learning rate


def build_model():
    """
    Build and compile the CNN model for vision-based Q-learning.

    Constructs a convolutional neural network that processes stacked camera frames
    and outputs Q-values for each action. Uses Huber loss for robustness to outliers
    and Adam optimizer for adaptive learning rates.

    Architecture:
        Input: (42, 42, 4) - 4 stacked grayscale frames
        ↓
        Conv2D: 32 filters, 4x4 kernel, stride 2, ReLU
        Output: (20, 20, 32)
        ↓
        Conv2D: 64 filters, 3x3 kernel, stride 1, ReLU
        Output: (18, 18, 64)
        ↓
        Flatten: 20736 units
        ↓
        Dense: 256 units, ReLU
        ↓
        Dense: 4 units, Linear (Q-values)

    Returns:
        tf.keras.models.Sequential: Compiled CNN model ready for training.
            - Input shape: (None, 42, 42, 4)
            - Output shape: (None, 4)
            - Total params: ~5.3M trainable parameters
            - Loss: Huber (smooth L1, robust to outliers)
            - Optimizer: Adam with learning_rate=0.0001

    Design Rationale:
        - **Conv layers**: Extract spatial features from images (edges, patterns)
        - **Stride 2 in first conv**: Aggressive downsampling for efficiency
        - **256 dense units**: Sufficient capacity for learning navigation policy
        - **Huber loss**: Less sensitive to large TD-errors than MSE
        - **Linear output**: Q-values can be positive or negative
    """
    model = Sequential()
    model.add(Conv2D(32, (4, 4), strides=(2, 2), activation="relu", input_shape=(42, 42, 4)))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(4, activation="linear"))
    model.compile(loss=Huber(), optimizer=Adam(learning_rate=LEARNING_RATE))
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
        trainer = TrainerDeepQLearningSimpleArena(
            model_name=MODEL_NAME,
            environment=environment,
            model=build_model(),
            memory_size=MEMORY_SIZE,
            gamma=GAMMA,
            epsilon=EPSILON,
            epsilon_decay=EPSILON_DECAY,
            batch_size=BATCH_SIZE,
            fit_frequency=FIT_FREQUENCY,
            update_target_model_frequency=UPDATE_TARGET_MODEL_FREQUENCY,
            per_alpha=PER_ALPHA,
            per_beta_start=PER_BETA_START,
        )
        trainer.run(EPOCHS)
        trainer.save_model()
    else:
        # Evaluation mode: Run single episode
        final_state = environment.run()
        logger().info(f"Final state: {final_state.to_json()}")

    # Cleanup
    environment.quit()
