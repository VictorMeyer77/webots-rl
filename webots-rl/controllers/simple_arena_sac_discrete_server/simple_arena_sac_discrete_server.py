# uncommented and unstable
import sys
from typing import Tuple

sys.path.append("webots-rl/libraries")

import logging

from brain.multi_trainer import MultiTrainer
from brain.multi_trainer.sac_discrete import TrainerSACDiscrete
from brain.utils.logger import logger
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Optimizer

# Model Configuration

MODEL_NAME = "simple_arena_sac_discrete"  # Base name for saved model files
NUM_ACTIONS = 3  # Number of discrete actions in the environment
FIT_STEP_FREQUENCY = 64  # Train model every N environment steps
GAMMA = 0.99  # Discount factor for future rewards [0,
ALPHA = 0.2  # Entropy temperature parameter
TAU = 0.005  # Target network update rate
BATCH_SIZE = 64  # Mini-batch size for training
GRAD_NORM_CLIP = 10.0  # Gradient norm clipping value
REPLAY_SIZE = 100000  # Replay buffer size
MIN_REPLAY_SIZE = 5000  # Minimum replay buffer size before training
ACTOR_LEARNING_RATE = 0.0003
CRITIC_LEARNING_RATE = 0.0003
TEMPERATURE_LEARNING_RATE = 0.0001
TARGET_ENTROPY_SCALE = -0.5

logger.add_console_logger(logging.INFO)
logger.add_file_logger(logging.INFO)


def build_model() -> Tuple[Model, Model, Model, Optimizer, Optimizer]:

    def encoder(inputs):
        x = Conv2D(32, (4, 4), strides=(2, 2), activation="relu")(inputs)
        x = Conv2D(64, (3, 3), strides=(1, 1), activation="relu")(x)
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        return x

    inputs = Input(shape=(42, 42, 4), name="obs")

    # Actor
    x_actor = encoder(inputs)
    policy_logits = Dense(NUM_ACTIONS, activation=None, name="policy_logits")(x_actor)
    actor = Model(inputs=inputs, outputs=policy_logits, name="sac_discrete_actor")

    # Critic 1
    x_q1 = encoder(inputs)
    q1_values = Dense(NUM_ACTIONS, activation=None, name="q1_values")(x_q1)
    critic1 = Model(inputs=inputs, outputs=q1_values, name="sac_discrete_critic1")

    # Critic 2
    x_q2 = encoder(inputs)
    q2_values = Dense(NUM_ACTIONS, activation=None, name="q2_values")(x_q2)
    critic2 = Model(inputs=inputs, outputs=q2_values, name="sac_discrete_critic2")

    actor_optimizer = Adam(learning_rate=ACTOR_LEARNING_RATE)
    critic_optimizer = Adam(learning_rate=CRITIC_LEARNING_RATE)

    return actor, critic1, critic2, actor_optimizer, critic_optimizer


def trainer(nb_env: int) -> MultiTrainer:

    actor, critic1, critic2, actor_optimizer, critic_optimizer = build_model()
    trainer_server = TrainerSACDiscrete(
        model_name=MODEL_NAME,
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        nb_env=nb_env,
        num_actions=NUM_ACTIONS,
        fit_step_frequency=FIT_STEP_FREQUENCY,
        gamma=GAMMA,
        alpha=ALPHA,
        tau=TAU,
        target_entropy_scale=TARGET_ENTROPY_SCALE,
        temperature_learning_rate=TEMPERATURE_LEARNING_RATE,
        batch_size=BATCH_SIZE,
        grad_norm_clip=GRAD_NORM_CLIP,
        replay_size=REPLAY_SIZE,
        min_replay_size=MIN_REPLAY_SIZE,
    )
    return trainer_server
