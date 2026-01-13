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
