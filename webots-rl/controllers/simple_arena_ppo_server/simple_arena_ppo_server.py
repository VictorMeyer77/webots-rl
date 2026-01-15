"""
Simple Arena PPO training server entry point.

This module serves as the main entry point for the PPO (Proximal Policy
Optimization) training server in the Simple Arena environment. It configures
and initializes the PPO trainer, defines the neural network architecture for
the actor-critic model, and coordinates multi-environment parallel training.

The server runs independently from Webots and communicates with multiple
environment instances via TCP sockets. It receives observations, computes
actions using the current policy, and performs PPO training updates on batches
of collected experience using the clipped surrogate objective.

PPO improves upon A2C by using a clipped surrogate objective that prevents
excessively large policy updates, making training more stable and sample-
efficient. The algorithm uses Generalized Advantage Estimation (GAE) for
variance reduction and performs multiple optimization epochs on each batch
of collected experience.

The actor-critic architecture uses a shared convolutional feature extractor
with separate heads for policy (actor) and value function (critic) outputs.
This approach enables efficient learning by sharing representations between
policy improvement and value estimation.

Constants:
    MODEL_NAME (str): Base name for saved model files ("simple_arena_ppo").
        A random suffix is appended during initialization for versioning.

    LEARNING_RATE (float): Learning rate for the Adam optimizer (0.0001).
        Lower values provide more stable but slower learning. PPO is less
        sensitive to learning rate than vanilla policy gradients due to the
        clipping mechanism.

    NUM_ACTIONS (int): Number of discrete actions in the action space (4).
        Typically corresponds to movement directions (forward, backward, left,
        right) for differential drive control.

    FIT_STEP_FREQUENCY (int): Number of environment steps to collect before
        performing a training update (128). This is the rollout length or
        batch size for PPO. Larger values provide more data but delay updates.

    GAMMA (float): Discount factor for future rewards (0.99). Values closer
        to 1 make the agent more far-sighted, considering long-term
        consequences. Standard range: [0.9, 0.999].

    VALUE_COEF (float): Coefficient for value loss in the total loss (0.25).
        Balances the importance of value function accuracy relative to policy
        improvement. Higher values (compared to A2C's 0.1) emphasize value
        learning for better advantage estimation.

    ENTROPY_COEF (float): Coefficient for entropy regularization (0.05).
        Encourages exploration by preventing premature convergence to
        deterministic policies. Higher than A2C's 0.01 to maintain exploration
        throughout training due to PPO's tendency to converge faster.

    GRAD_NORM_CLIP (float): Maximum norm for gradient clipping (0.5).
        Prevents excessively large gradient updates that can destabilize
        training. Essential for stable PPO training.

    LAMBDA (float): GAE (Generalized Advantage Estimation) lambda parameter
        (0.95). Controls the bias-variance tradeoff in advantage estimation:
        - λ=0: High bias, low variance (one-step TD)
        - λ=1: Low bias, high variance (Monte Carlo)
        Standard range: [0.9, 0.99].

    PPO_EPOCHS (int): Number of optimization epochs per training update (4).
        PPO performs multiple passes over the collected batch of experience,
        reusing data for better sample efficiency. More epochs improve learning
        but increase computation time. Standard range: [3, 10].

    MINI_BATCH_SIZE (int): Size of mini-batches for gradient updates (32).
        Each PPO epoch divides the rollout buffer into mini-batches for
        stochastic gradient descent. Smaller batches provide noisier but more
        frequent updates. Must divide FIT_STEP_FREQUENCY evenly.

    CLIP_RATIO (float): PPO clipping parameter epsilon (0.2). Determines the
        maximum allowed policy update magnitude. The clipped objective prevents
        policy updates larger than [1-ε, 1+ε] times the old policy probability
        ratio. Standard range: [0.1, 0.3].
"""

import sys

sys.path.append("webots-rl/libraries")

import logging

from brain.multi_trainer import MultiTrainer
from brain.multi_trainer.ppo import TrainerPPO
from brain.utils.logger import logger
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Optimizer

# Model Configuration
MODEL_NAME = "simple_arena_ppo"  # Base name for saved model files
LEARNING_RATE = 0.0001  # Adam optimizer learning rate
NUM_ACTIONS = 4  # Number of possible actions in the environment
FIT_STEP_FREQUENCY = 128  # Train model every N environment steps
GAMMA = 0.99  # Discount factor for future rewards [0, 1]
VALUE_COEF = 0.25  # Value loss coefficient
ENTROPY_COEF = 0.05  # Entropy regularization coefficient
GRAD_NORM_CLIP = 0.5  # Gradient norm clipping value
LAMBDA = 0.95  # GAE lambda parameter
PPO_EPOCHS = 4  # Number of PPO epochs per update
MINI_BATCH_SIZE = 32  # Mini-batch size for PPO updates
CLIP_RATIO = 0.2  # PPO clipping ratio


logger.add_console_logger(logging.INFO)
logger.add_file_logger(logging.INFO)


def build_model() -> (Model, Optimizer):
    """
    Constructs the actor-critic neural network architecture for PPO.

    This function builds a convolutional neural network with shared feature
    extraction layers and separate heads for policy (actor) and value function
    (critic) outputs. The architecture is identical to the A2C model but
    optimized for PPO's training dynamics, which require more stable value
    estimates due to multiple optimization epochs on the same data.

    Architecture:
        Input Layer:
            - Shape: (42, 42, 4)
            - 42x42 pixel grayscale images with 4 stacked frames for temporal
              context and motion information
            - Stacked frames help the policy learn velocity and acceleration

        Convolutional Layers:
            - Conv2D(32, kernel=4x4, stride=2, activation=ReLU)
              * Extracts low-level visual features (edges, corners, textures)
              * Stride 2 provides spatial downsampling for computational efficiency
              * Output shape: (20, 20, 32)
              * Receptive field: 4x4 pixels

            - Conv2D(64, kernel=3x3, stride=1, activation=ReLU)
              * Extracts higher-level visual features (patterns, shapes, objects)
              * Stride 1 preserves spatial resolution for fine-grained features
              * Output shape: (18, 18, 64)
              * Receptive field: 10x10 pixels (cumulative)

        Flatten Layer:
            - Converts 2D feature maps to 1D vector
            - Output shape: (20736,) = 18 * 18 * 64
            - Prepares spatial features for fully connected layers

        Fully Connected Layer:
            - Dense(256, activation=ReLU)
            - Shared representation for both actor and critic heads
            - Learns high-level abstract features for decision-making
            - Output shape: (256,)

        Actor Head (Policy Network):
            - Dense(4, activation=None)
            - Outputs raw logits for 4 discrete actions (no softmax)
            - Softmax is applied during action sampling or loss computation
            - Linear activation allows unbounded logits for stable training
            - Output shape: (4,)
            - Actions are sampled using: π(a|s) = softmax(logits)

        Critic Head (Value Function Network):
            - Dense(1, activation=None)
            - Outputs state value estimate V(s)
            - Linear activation for unbounded value predictions
            - Output shape: (1,)
            - Used for computing advantages: A(s,a) = Q(s,a) - V(s)

    The shared convolutional layers enable efficient feature learning by
    leveraging the correlation between policy and value function. Both heads
    benefit from the same visual representations, reducing training time and
    improving sample efficiency. This is particularly important for PPO, which
    performs multiple optimization passes over the same data.

    Returns:
        tuple[Model, Optimizer]: A tuple containing:
            - Model: Compiled Keras model with two outputs [policy_logits, value].
              The model takes stacked frame observations as input and produces
              action logits and state value estimates. Both outputs are used
              during PPO training for computing policy loss and value loss.

            - Optimizer: Adam optimizer configured with the specified learning
              rate (0.0001). Adam provides adaptive learning rates and momentum
              for stable convergence. The optimizer state is preserved across
              PPO epochs for efficient training.

    Model Signature:
        inputs: (batch_size, 42, 42, 4) - Stacked grayscale observations
        outputs:
            - policy_logits: (batch_size, 4) - Raw action logits
            - value: (batch_size, 1) - State value estimates
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
    Initializes and configures the PPO trainer for multi-environment training.

    This function creates a PPO trainer instance with the specified neural
    network architecture, optimizer, and hyperparameters. The trainer manages
    parallel training across multiple environment instances, collecting
    experience via TCP sockets and performing PPO training updates with the
    clipped surrogate objective and Generalized Advantage Estimation (GAE).

    The trainer uses the Proximal Policy Optimization algorithm, which improves
    upon A2C by:
    - Using a clipped surrogate objective to prevent large policy updates
    - Performing multiple optimization epochs on each batch of experience
    - Employing GAE for better advantage estimation with bias-variance control
    - Using mini-batches for stochastic gradient descent

    Args:
        nb_env (int): Number of parallel environments to train with. More
            environments provide more diverse experience and faster data
            collection, but require more computational resources and network
            bandwidth. Typical values: 4-16 environments.

            Considerations:
            - Each environment collects FIT_STEP_FREQUENCY steps per update
            - Total batch size = nb_env * FIT_STEP_FREQUENCY
            - More environments reduce correlation in experience
            - Optimal number depends on hardware and task complexity

    Returns:
        MultiTrainer: Configured TrainerPPO instance ready for training.
            The trainer provides the following capabilities:
            - TCP socket communication with environment agents
            - Experience replay buffer management with trajectory storage
            - GAE computation for advantage estimation
            - PPO loss computation with clipped objective
            - Multiple optimization epochs with mini-batch SGD
            - Gradient clipping for training stability
            - TensorBoard logging for training metrics
            - Periodic model checkpointing with versioning
    """
    model, optimizer = build_model()
    trainer_server = TrainerPPO(
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
        lambda_=LAMBDA,
        ppo_epochs=PPO_EPOCHS,
        mini_batch_size=MINI_BATCH_SIZE,
        clip_ratio=CLIP_RATIO,
    )
    return trainer_server
