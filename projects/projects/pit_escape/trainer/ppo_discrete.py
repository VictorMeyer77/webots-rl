import logging

import numpy as np
from numpy.typing import NDArray
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from corl.model.discrete.actor_critic import ModelActorCritic
from corl.schemas.learning import Observation
from corl.schemas.tracker import StepKey
from corl.trainer.algorithm.discrete.ppo import TrainerPPO
from corl.utils.config import Config
from corl.utils.logger import setup_logging

ACTION_SIZE = 9  # Number of discrete actions available to the agent
TRANSITIONS = 2_000_000  # Total number of training transitions
MODEL_CHECKPOINT_FREQUENCY = 250_000  # Save a checkpoint every N transitions
GAMMA = 0.99  # Discount factor: how much future rewards are valued
ENTROPY_COEFF = 0.02  # Entropy bonus weight
VALUE_LOSS_COEFF = 0.5  # Critic loss scaling factor
ACTOR_LR = 3e-4  # Actor Adam learning rate
CRITIC_LR = 1e-4  # Critic Adam learning rate
CLIP_RANGE = 0.2  # PPO surrogate objective clipping parameter ε
PPO_EPOCHS = 4  # Number of optimisation passes over each collected rollout
MINI_BATCH_SIZE = 64  # Number of transitions per mini-batch within each PPO epoch
UPDATE_FREQUENCY = 512  # Transitions to collect between PPO update cycles
MAX_GRAD_NORM = 0.5  # Maximum L2 norm for gradient clipping
INPUT_SHAPE = 18  # Flat sensor vector size


def build_actor() -> Sequential:
    """
    Build the actor network.

    Maps flat sensor observations to action logits (pre-softmax). The softmax
    sampling is applied inside :class:`~corl.model.actor_critic.ModelActorCritic`.

    Returns:
        Keras ``Sequential`` model outputting ``ACTION_SIZE`` logits.
    """
    model = Sequential(
        [
            Dense(128, activation="relu", input_shape=(INPUT_SHAPE,)),
            Dense(64, activation="relu"),
            Dense(ACTION_SIZE, activation="linear"),
        ]
    )
    model.summary()
    return model


def build_critic() -> Sequential:
    """
    Build the critic network.

    Maps flat sensor observations to a scalar state-value estimate.

    Returns:
        Keras ``Sequential`` model outputting a single value.
    """
    model = Sequential(
        [
            Dense(128, activation="relu", input_shape=(INPUT_SHAPE,)),
            Dense(64, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )
    model.summary()
    return model


class PitEscapePPODiscrete(TrainerPPO):
    """
    PPO trainer for the pit escape task.

    Uses separate actor and critic networks with dense layers over the flat
    sensor observation vector. The clipped surrogate objective with multiple
    mini-batch epochs stabilises training on the low-dimensional input.
    """

    def parse_observations(
        self, observations: list[tuple[StepKey, Observation]]
    ) -> list[tuple[StepKey, NDArray[np.float32]]]:
        """
        Convert raw observations to float32 arrays for all workers.

        Args:
            observations: Raw observations received from all workers in the
                current rollout.

        Returns:
            List of ``(StepKey, flat_obs)`` pairs for every worker, where each
            observation is a float32 array of shape ``(INPUT_SHAPE,)``.
        """
        return [
            (
                step_key,
                np.asarray(observation.data["base"], dtype=np.float32),
            )
            for step_key, observation in observations
        ]

    def params(self) -> dict[str, str | int | float]:
        """
        Return hyperparameters logged to the experiment tracker.

        Extends the parent params with the input shape specific to this trainer.

        Returns:
            Dict mapping parameter names to their values.
        """
        return super().params() | {"input_shape": INPUT_SHAPE}


if __name__ == "__main__":
    config = Config()
    setup_logging(config)
    logger = logging.getLogger(__name__)

    model = ModelActorCritic(
        actor=build_actor(),
        critic=build_critic(),
        action_size=ACTION_SIZE,
    )

    PitEscapePPODiscrete(
        config=config,
        model=model,
        model_checkpoint_frequency=MODEL_CHECKPOINT_FREQUENCY,
        gamma=GAMMA,
        clip_range=CLIP_RANGE,
        ppo_epochs=PPO_EPOCHS,
        mini_batch_size=MINI_BATCH_SIZE,
        entropy_coeff=ENTROPY_COEFF,
        value_loss_coeff=VALUE_LOSS_COEFF,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        update_frequency=UPDATE_FREQUENCY,
        max_grad_norm=MAX_GRAD_NORM,
    ).run(epochs=TRANSITIONS)
