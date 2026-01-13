import json
import time

import brain.utils.tcp_socket as tcp
import numpy as np
import tensorflow as tf
from brain.multi_trainer import MultiTrainer
from brain.utils.logger import logger

MODEL_SAVE_FREQUENCY_MINUTES = 10


class TrainerA2C(MultiTrainer):

    num_actions: int
    fit_step_frequency: int
    gamma: float
    entropy_coefficient: float
    value_loss_coefficient: float
    grad_norm_clip: float
    step_count: int = 0

    def __init__(
        self,
        model_name: str,
        model: tf.keras.models.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        nb_env: int,
        num_actions: int,
        fit_step_frequency: int,
        gamma: float,
        entropy_coefficient: float,
        value_loss_coefficient: float,
        grad_norm_clip: float,
    ):

        super().__init__(
            model_name=model_name, model=model, optimizer=optimizer, nb_env=nb_env, memory_size=fit_step_frequency * 2
        )
        self.num_actions = num_actions
        self.fit_step_frequency = fit_step_frequency
        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient
        self.value_loss_coefficient = value_loss_coefficient
        self.grad_norm_clip = grad_norm_clip
        self.step_count = 0
        self.last_model_save_time = time.time()

    def policy(self, observation: np.ndarray) -> (int, float):
        observation = np.expand_dims(observation, axis=0)
        logits, value = self.model(observation)
        action = tf.random.categorical(logits, 1)
        action = tf.squeeze(action, axis=1)
        return int(action.numpy()), float(value.numpy())

    def compute_returns(self, rewards: np.ndarray, dones: np.ndarray, last_values: np.ndarray) -> np.ndarray:
        returns = []
        r = last_values
        for t in reversed(range(len(rewards))):
            r = rewards[t] + self.gamma * r * (1 - dones[t])
            returns.insert(0, r)
        return np.array(returns)

    def gather_batch(self):
        ports = sorted(self.memory.keys())
        steps = self.memory_count()

        observations, actions, rewards, dones, values = [], [], [], [], []

        for step in range(steps):
            step_observation, step_action, step_reward, step_done, step_value = [], [], [], [], []

            for port in ports:
                observation, action, reward, done, value = self.memory[port].popleft()
                step_observation.append(observation)
                step_action.append(action)
                step_reward.append(reward)
                step_done.append(done)
                step_value.append(value)

            observations.append(step_observation)
            actions.append(step_action)
            rewards.append(step_reward)
            dones.append(step_done)
            values.append(step_value)

        return (
            np.array(observations, dtype=np.float32),  # (steps, num_env, obs_dim)
            np.array(actions, dtype=np.int32),  # (steps, num_env)
            np.array(rewards, dtype=np.float32),  # (steps, num_env)
            np.array(dones, dtype=np.float32),  # (steps, num_env)
            np.array(values, dtype=np.float32),  # (steps, num_env)
        )

    def fit_model(self) -> None:
        observations, actions, rewards, dones, values = self.gather_batch()

        _, last_values = self.model(tf.convert_to_tensor(observations[-1], dtype=tf.float32))
        last_values = tf.squeeze(last_values).numpy()

        returns = self.compute_returns(rewards, dones, last_values)

        steps, num_env, height, width, frames = observations.shape
        observation_batch = observations.reshape(steps * num_env, height, width, frames)
        actions_batch = actions.reshape(steps * num_env)
        returns_batch = tf.convert_to_tensor(returns.reshape(steps * num_env), dtype=tf.float32)
        values_batch = values.reshape(steps * num_env)
        advantages = returns_batch - values_batch
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

        with tf.GradientTape() as tape:
            logits, predicted_values = self.model(observation_batch)
            predicted_values = tf.squeeze(predicted_values, axis=-1)

            action_masks = tf.one_hot(actions_batch, self.num_actions)
            log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)

            policy_loss = -tf.reduce_mean(log_probs * advantages)
            value_loss = tf.reduce_mean(tf.square(returns_batch - predicted_values))
            pi = tf.nn.softmax(logits)
            entropy = -tf.reduce_mean(tf.reduce_sum(pi * tf.math.log(pi + 1e-8), axis=1))

            loss = policy_loss + value_loss * self.value_loss_coefficient - entropy * self.entropy_coefficient

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.grad_norm_clip)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        with self.tb_writer.as_default():
            tf.summary.scalar("A2C/Loss", loss, step=self.step_count)
            tf.summary.scalar("A2C/Policy_Loss", policy_loss, step=self.step_count)
            tf.summary.scalar("A2C/Value_Loss", value_loss, step=self.step_count)
            tf.summary.scalar("A2C/Entropy", entropy, step=self.step_count)
            tf.summary.scalar("A2C/Reward_Mean", tf.reduce_mean(rewards), step=self.step_count)
            tf.summary.scalar("A2C/Advantage_Mean", tf.reduce_mean(advantages), step=self.step_count)

        self.step_count += 1

    def run(self) -> None:

        self.accept_connections()

        sockets = self.sockets

        while True:

            closed_sockets = []

            for conn, port in sockets:
                msg = tcp.read(conn)

                if msg is not None:
                    obj = json.loads(msg)
                    message_type = obj["message_type"]

                    if message_type == "policy":
                        observation = np.array(obj["observation"])
                        action, value = self.policy(observation)
                        tcp.send(conn, json.dumps({"action": action, "value": value}))

                    elif message_type == "step":
                        self.memory_append(
                            port, np.array(obj["observation"]), obj["action"], obj["reward"], obj["done"], obj["value"]
                        )

                    elif message_type == "terminated":
                        closed_sockets.append(port)
                        logger().info(f"Environment on port {port} has terminated.")

            sockets = [s for s in sockets if s[1] not in closed_sockets]

            if len(sockets) == 0:
                self.save_model()
                break

            if self.memory_count() == self.fit_step_frequency:
                self.fit_model()

            if self.step_count > 0 and time.time() - self.last_model_save_time >= MODEL_SAVE_FREQUENCY_MINUTES * 60:
                self.save_model()
                self.last_model_save_time = time.time()
