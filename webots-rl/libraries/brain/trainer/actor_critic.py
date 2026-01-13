import os
from abc import abstractmethod

import numpy as np
import tensorflow as tf
from brain.environment import Environment
from brain.model import MODEL_PATH
from brain.trainer import Trainer
from brain.utils.logger import logger

MODEL_SAVE_FREQUENCY = 500  # Save model checkpoint every N epochs


class TrainerActorCritic(Trainer):

    model: tf.keras.models.Model | None
    gamma: float
    optimizer: tf.keras.optimizers.Optimizer | None
    episode_metrics: dict[str, list[float]]

    def __init__(
        self,
        environment: Environment,
        model_name: str,
        model: tf.keras.models.Model,
        gamma: float,
        optimizer: tf.keras.optimizers.Optimizer,
    ):

        super().__init__(environment=environment, model_name=model_name)
        self.model = model
        self.gamma = gamma
        self.optimizer = optimizer
        self._init_episode_metrics()

    def _init_episode_metrics(self) -> None:
        self.episode_metrics = {
            "actor_loss": [],
            "critic_loss": [],
            "td_error": [],
        }

    def fit_model(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:

        with tf.GradientTape() as tape:
            obs_logits, obs_value = self.model(observation)
            _, next_obs_value = self.model(next_observation)

            obs_value = tf.squeeze(obs_value)
            next_obs_value = tf.squeeze(next_obs_value)

            td_target = reward + self.gamma * next_obs_value * (1.0 - float(terminated))
            td_error = td_target - obs_value

            action_mask = tf.one_hot([action], obs_logits.shape[-1])
            log_probs = tf.nn.log_softmax(obs_logits)
            log_prob_action = tf.reduce_sum(log_probs * action_mask, axis=-1)
            actor_loss = -log_prob_action * tf.stop_gradient(td_error)

            critic_loss = tf.square(td_error)

            actor_loss = tf.reduce_mean(actor_loss)
            critic_loss = tf.reduce_mean(critic_loss)

            combined_loss = actor_loss + critic_loss

        grads = tape.gradient(combined_loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.episode_metrics["actor_loss"].append(float(actor_loss.numpy()))
        self.episode_metrics["critic_loss"].append(float(critic_loss.numpy()))
        self.episode_metrics["td_error"].append(float(td_error.numpy()))

    def run(self, epochs: int) -> None:

        for epoch in range(epochs):

            reward = self.simulation()

            with self.tb_writer.as_default():
                tf.summary.scalar("ActorCritic/ActorLoss", np.sum(self.episode_metrics["actor_loss"]), epoch)
                tf.summary.scalar("ActorCritic/CriticLoss", np.sum(self.episode_metrics["critic_loss"]), epoch)
                tf.summary.scalar("ActorCritic/TD_Error", np.sum(self.episode_metrics["td_error"]), epoch)
                tf.summary.scalar("ActorCritic/Reward", reward, epoch)

            self._init_episode_metrics()
            self.environment.reset()

            logger().info(f"Epoch {epoch + 1}/{epochs} completed with reward {reward}")

            if epoch % MODEL_SAVE_FREQUENCY == 0 and epoch != 0:
                self.save_model()

        self.close_tb()

    def save_model(self) -> None:

        path = os.path.join(MODEL_PATH, self.model_name + ".keras")
        self.model.save(path)
        logger().info(f"Model saved successfully at {path}")

    @abstractmethod
    def simulation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        raise NotImplementedError("Method simulation() not implemented in TrainerReinforce.")
