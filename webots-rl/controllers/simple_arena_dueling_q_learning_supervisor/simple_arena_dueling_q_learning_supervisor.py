"""
Supervisor Script for Dueling Deep Q-Learning Vision-Based Navigation Training.

Main entry point for training or evaluating a Dueling Deep Q-Learning agent in the simple
arena environment. The supervisor coordinates the training process, manages the environment
state, and controls episode resets.

Dueling Architecture:
    This implementation uses Dueling DQN (Wang et al., 2016), which separates the Q-value
    estimation into two streams:
    - Value Stream: V(s) - estimates how good it is to be in a state
    - Advantage Stream: A(s,a) - estimates the advantage of each action
    - Combined: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))

    This decomposition provides several benefits:
    - Better generalization: Value stream learns state quality independently of actions
    - Faster learning: Network can evaluate states without learning every action
    - Improved stability: Reduces overestimation compared to standard DQN

Modes:
    - Training (TRAIN=1): Runs RL training loop for specified epochs
    - Evaluation (TRAIN=0): Runs single episode with trained model for testing

Key Hyperparameters:
    - EPOCHS: 2000 training episodes
    - MEMORY_SIZE: 100k experiences in prioritized replay buffer
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
from brain.utils.register_tf import dueling_combine_streams
from controller import Supervisor
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Model Configuration
MODEL_NAME = "simple_arena_dueling_q_learning"  # Base name for saved model files

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


def build_model() -> Model:
    """
    Builds a dueling deep Q-network for vision-based navigation.

    Implements the Dueling DQN architecture (Wang et al., 2016) which separates
    the representation of state values and action advantages. This decomposition
    allows the network to learn which states are valuable independent of the
    action choice, leading to better policy evaluation and faster learning.

    Architecture:
        Input: (42, 42, 4) stacked grayscale camera frames

        Feature Extraction:
            - Conv2D(32, 4x4, stride=2) + ReLU
            - Conv2D(64, 3x3, stride=1) + ReLU
            - Flatten
            - Dense(512) + ReLU

        Value Stream (learns state value V(s)):
            - Dense(256) + ReLU
            - Dense(1) linear → V(s)

        Advantage Stream (learns action advantages A(s,a)):
            - Dense(256) + ReLU
            - Dense(4) linear → A(s,a) for each action

        Combination Layer (Lambda):
            Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            Uses dueling_combine_streams function from brain.utils.register_tf

    Custom Objects:
        The Lambda layer uses dueling_combine_streams, which is registered via
        @keras.saving.register_keras_serializable() in brain.utils.register_tf.
        This ensures the model can be saved and loaded correctly without
        deserialization errors.

    Compilation:
        - Loss: Huber (robust to outliers, reduces impact of large TD-errors)
        - Optimizer: Adam with learning_rate=0.0001
        - Output: 4 Q-values (one per action: forward, left, right, backward)

    Returns:
        tf.keras.Model: Compiled dueling Q-network ready for training
    """

    inputs = Input(shape=(42, 42, 4))
    x = Conv2D(32, (4, 4), strides=(2, 2), activation="relu")(inputs)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation="relu")(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)

    value_stream = Dense(256, activation="relu")(x)
    value = Dense(1, activation="linear")(value_stream)

    advantage_stream = Dense(256, activation="relu")(x)
    advantage = Dense(4, activation="linear")(advantage_stream)

    q_values = Lambda(dueling_combine_streams)([value, advantage])
    model = Model(inputs=inputs, outputs=q_values)
    model.compile(loss=Huber(), optimizer=Adam(learning_rate=LEARNING_RATE))
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
