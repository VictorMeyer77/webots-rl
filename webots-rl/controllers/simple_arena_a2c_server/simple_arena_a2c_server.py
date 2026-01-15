"""
Simple Arena A2C training server entry point.

This module serves as the main entry point for the A2C (Advantage Actor-Critic)
training server in the Simple Arena environment. It configures and initializes
the A2C trainer, defines the neural network architecture for the actor-critic
model, and coordinates multi-environment parallel training.

The server runs independently from Webots and communicates with multiple
environment instances via TCP sockets. It receives observations, computes
actions using the current policy, and performs A2C training updates on batches
of collected experience.

The actor-critic architecture uses a shared convolutional feature extractor
with separate heads for policy (actor) and value function (critic) outputs.
This approach enables efficient learning by sharing representations between
policy improvement and value estimation.

Constants:
    MODEL_NAME (str): Base name for saved model files ("simple_arena_a2c").
        A random suffix is appended during initialization for versioning.

    LEARNING_RATE (float): Learning rate for the Adam optimizer (0.0001).
        Lower values provide more stable but slower learning.

    NUM_ACTIONS (int): Number of discrete actions in the action space (4).
        Typically corresponds to movement directions (forward, backward, left, right).

    FIT_STEP_FREQUENCY (int): Number of environment steps to collect before
        performing a training update (16). Controls the batch size for A2C updates.

    GAMMA (float): Discount factor for future rewards (0.99). Values closer to 1
        make the agent more far-sighted, considering long-term consequences.

    VALUE_COEF (float): Coefficient for value loss in the total loss (0.1).
        Balances the importance of value function accuracy relative to policy
        improvement. Lower values emphasize policy learning.

    ENTROPY_COEF (float): Coefficient for entropy regularization (0.01).
        Encourages exploration by preventing premature convergence to
        deterministic policies. Higher values increase exploration.

    GRAD_NORM_CLIP (float): Maximum norm for gradient clipping (0.5).
        Prevents excessively large gradient updates that can destabilize training.
"""

import sys

sys.path.append("webots-rl/libraries")

import logging

from brain.multi_trainer import MultiTrainer
from brain.multi_trainer.a2c import TrainerA2C
from brain.utils.logger import logger
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Optimizer

# Model Configuration

MODEL_NAME = "simple_arena_a2c"  # Base name for saved model files
LEARNING_RATE = 0.0001  # Adam optimizer learning rate
NUM_ACTIONS = 4  # Number of possible actions in the environment
FIT_STEP_FREQUENCY = 16  # Train model every N environment steps
GAMMA = 0.99  # Discount factor for future rewards [0, 1]
VALUE_COEF = 0.1  # Value loss coefficient
ENTROPY_COEF = 0.01  # Entropy regularization coefficient
GRAD_NORM_CLIP = 0.5  # Gradient norm clipping value

logger.add_console_logger(logging.INFO)
logger.add_file_logger(logging.INFO)


def build_model() -> (Model, Optimizer):
    """
    Constructs the actor-critic neural network architecture for A2C.

    This function builds a convolutional neural network with shared feature
    extraction layers and separate heads for policy (actor) and value function
    (critic) outputs. The architecture is designed for processing stacked
    grayscale camera observations (4 frames) from the Simple Arena environment.

    Architecture:
        Input Layer:
            - Shape: (42, 42, 4)
            - 42x42 pixel grayscale images with 4 stacked frames for temporal context

        Convolutional Layers:
            - Conv2D(32, kernel=4x4, stride=2, activation=ReLU)
              * Extracts low-level visual features (edges, corners)
              * Stride 2 provides spatial downsampling for efficiency
              * Output shape: (20, 20, 32)

            - Conv2D(64, kernel=3x3, stride=1, activation=ReLU)
              * Extracts higher-level visual features (patterns, shapes)
              * Stride 1 preserves spatial resolution
              * Output shape: (18, 18, 64)

        Flatten Layer:
            - Converts 2D feature maps to 1D vector
            - Output shape: (20736,)

        Fully Connected Layer:
            - Dense(256, activation=ReLU)
            - Shared representation for both actor and critic
            - Output shape: (256,)

        Actor Head (Policy):
            - Dense(4, activation=None)
            - Outputs raw logits for 4 discrete actions (no softmax)
            - Softmax is applied during action sampling or loss computation
            - Output shape: (4,)

        Critic Head (Value Function):
            - Dense(1, activation=None)
            - Outputs state value estimate V(s)
            - Linear activation for unbounded value predictions
            - Output shape: (1,)

    The shared convolutional layers enable efficient feature learning by
    leveraging the correlation between policy and value function. Both heads
    benefit from the same visual representations, reducing training time and
    improving sample efficiency.

    Returns:
        tuple[Model, Optimizer]: A tuple containing:
            - Model: Compiled Keras model with two outputs [policy_logits, value].
              The model takes stacked frame observations as input and produces
              action logits and state value estimates.
            - Optimizer: Adam optimizer configured with the specified learning
              rate. Adam provides adaptive learning rates and momentum for
              stable convergence.
    """
    inputs = Input(shape=(42, 42, 4))

    x = Conv2D(32, (4, 4), strides=(2, 2), activation="relu")(inputs)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation="relu")(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)

    # Actor
    policy_logits = Dense(4, activation=None)(x)

    # Critic
    value = Dense(1, activation=None)(x)

    model = Model(inputs=inputs, outputs=[policy_logits, value])

    optimizer = Adam(learning_rate=LEARNING_RATE)

    return model, optimizer


def trainer(nb_env: int) -> MultiTrainer:
    """
    Initializes and configures the A2C trainer for multi-environment training.

    This function creates an A2C trainer instance with the specified neural
    network architecture, optimizer, and hyperparameters. The trainer manages
    parallel training across multiple environment instances, collecting
    experience via TCP sockets and performing synchronous A2C updates.

    The trainer uses the Advantage Actor-Critic algorithm, which combines:
    - Policy gradient methods for policy improvement (actor)
    - Temporal difference learning for value estimation (critic)
    - Advantage estimation for variance reduction

    Args:
        nb_env (int): Number of parallel environments to train with. More
            environments provide more diverse experience and faster data
            collection, but require more computational resources. Typical
            values: 4-16 environments.

    Returns:
        MultiTrainer: Configured TrainerA2C instance ready for training.
            The trainer provides the following capabilities:
            - TCP socket communication with environment agents
            - Experience replay memory management
            - A2C loss computation and gradient updates
            - TensorBoard logging for training metrics
            - Periodic model checkpointing
    """
    model, optimizer = build_model()
    trainer_server = TrainerA2C(
        model_name=MODEL_NAME,
        model=model,
        optimizer=optimizer,
        nb_env=nb_env,
        num_actions=NUM_ACTIONS,
        fit_step_frequency=FIT_STEP_FREQUENCY,
        gamma=GAMMA,
        entropy_coefficient=ENTROPY_COEF,
        value_loss_coefficient=VALUE_COEF,
        grad_norm_clip=GRAD_NORM_CLIP,
    )
    return trainer_server
