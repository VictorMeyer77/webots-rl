
import sys
from typing import Tuple

sys.path.append("webots-rl/libraries")

import logging

from brain.multi_trainer import MultiTrainer
from brain.multi_trainer.a2c import TrainerA2C
from brain.utils.logger import logger
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Optimizer
import tensorflow as tf

# Model Configuration

MODEL_NAME = "simple_arena_sac"  # Base name for saved model files

NUM_ACTIONS = 4  # Number of discrete actions in the environment
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.0001

logger.add_console_logger(logging.INFO)
logger.add_file_logger(logging.INFO)


def build_model() -> Tuple[Model, Model, Model, Model, Model, Optimizer, Optimizer]:


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

    # Targets (copies)
    critic1_target = tf.keras.models.clone_model(critic1)
    critic1_target.set_weights(critic1.get_weights())
    critic1_target.trainable = False

    critic2_target = tf.keras.models.clone_model(critic2)
    critic2_target.set_weights(critic2.get_weights())
    critic2_target.trainable = False

    actor_optimizer = Adam(learning_rate=ACTOR_LEARNING_RATE)
    critic_optimizer = Adam(learning_rate=CRITIC_LEARNING_RATE)

    return actor, critic1, critic2, critic1_target, critic2_target, actor_optimizer, critic_optimizer

def trainer(nb_env: int) -> MultiTrainer:

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
