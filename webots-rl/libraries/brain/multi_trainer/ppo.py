import json
import time

import brain.utils.tcp_socket as tcp
import numpy as np
import tensorflow as tf
from brain.multi_trainer import MultiTrainer
from brain.utils.logger import logger

MODEL_SAVE_FREQUENCY_MINUTES = 10


class TrainerPPO(MultiTrainer):

    num_actions: int
    fit_step_frequency: int
    gamma: float
    entropy_coefficient: float
    value_loss_coefficient: float
    grad_norm_clip: float
    lambda_: float
    ppo_epochs: int
    mini_batch_size: int
    clip_ratio: float
    step_count: int = 0

    def __init__(
        self,
        model_name: str,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        nb_env: int,
        num_actions: int,
        fit_step_frequency: int,
        gamma: float,
        entropy_coefficient: float,
        value_loss_coefficient: float,
        grad_norm_clip: float,
        lambda_: float,
        ppo_epochs: int,
        mini_batch_size: int,
        clip_ratio: float,
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
        self.lambda_ = lambda_
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.clip_ratio = clip_ratio
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

    def compute_gae(
        self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, next_value: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

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

        # Get next state values for proper bootstrapping
        _, last_values = self.model(tf.convert_to_tensor(observations[-1], dtype=tf.float32))
        last_values = tf.squeeze(last_values).numpy()

        # Bootstrap properly: mask terminal states
        next_values = last_values * (1 - dones[-1])

        # Use GAE instead of simple advantage estimation
        steps, num_env = rewards.shape
        advantages_gae = []
        returns_gae = []

        # Compute GAE for each environment separately
        for env_idx in range(num_env):
            adv, ret = self.compute_gae(
                rewards[:, env_idx], values[:, env_idx], dones[:, env_idx], next_values[env_idx]
            )
            advantages_gae.append(adv)
            returns_gae.append(ret)

        advantages_gae = np.array(advantages_gae).T  # (steps, num_env)
        returns_gae = np.array(returns_gae).T  # (steps, num_env)

        # Reshape for training
        _, _, height, width, frames = observations.shape
        observation_batch = observations.reshape(steps * num_env, height, width, frames)
        actions_batch = actions.reshape(steps * num_env)
        returns_batch = tf.convert_to_tensor(returns_gae.reshape(steps * num_env), dtype=tf.float32)
        advantages = advantages_gae.reshape(steps * num_env)

        # Normalize advantages
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

        # Compute old log probabilities (for PPO ratio)
        old_logits, _ = self.model(observation_batch)
        action_masks = tf.one_hot(actions_batch, self.num_actions)
        old_log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(old_logits), axis=1)
        old_log_probs = tf.stop_gradient(old_log_probs)  # Don't backprop through old policy

        batch_size = observation_batch.shape[0]
        indices = np.arange(batch_size)

        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mini_batch_indices = indices[start:end]

                obs_mini = tf.gather(observation_batch, mini_batch_indices)
                actions_mini = tf.gather(actions_batch, mini_batch_indices)
                returns_mini = tf.gather(returns_batch, mini_batch_indices)
                advantages_mini = tf.gather(advantages, mini_batch_indices)
                old_log_probs_mini = tf.gather(old_log_probs, mini_batch_indices)

                with tf.GradientTape() as tape:
                    logits, predicted_values = self.model(obs_mini)
                    predicted_values = tf.squeeze(predicted_values, axis=-1)

                    # Compute new log probabilities
                    action_masks_mini = tf.one_hot(actions_mini, self.num_actions)
                    new_log_probs = tf.reduce_sum(action_masks_mini * tf.nn.log_softmax(logits), axis=1)

                    # PPO clipped surrogate loss
                    ratio = tf.exp(new_log_probs - old_log_probs_mini)
                    clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                    policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages_mini, clipped_ratio * advantages_mini))

                    # Value loss
                    value_loss = tf.reduce_mean(tf.square(returns_mini - predicted_values))

                    # Entropy bonus
                    pi = tf.nn.softmax(logits)
                    entropy = -tf.reduce_mean(tf.reduce_sum(pi * tf.math.log(pi + 1e-8), axis=1))

                    loss = policy_loss + value_loss * self.value_loss_coefficient - entropy * self.entropy_coefficient

                grads = tape.gradient(loss, self.model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, self.grad_norm_clip)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Log metrics (once per fit_model call)
        with self.tb_writer.as_default():
            tf.summary.scalar("PPO/Loss", loss, step=self.step_count)
            tf.summary.scalar("PPO/Policy_Loss", policy_loss, step=self.step_count)
            tf.summary.scalar("PPO/Value_Loss", value_loss, step=self.step_count)
            tf.summary.scalar("PPO/Entropy", entropy, step=self.step_count)
            tf.summary.scalar("PPO/Reward_Mean", tf.reduce_mean(rewards), step=self.step_count)
            tf.summary.scalar("PPO/Advantage_Mean", tf.reduce_mean(advantages), step=self.step_count)
            tf.summary.scalar(
                "PPO/Clip_Fraction",
                tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > self.clip_ratio, tf.float32)),
                step=self.step_count,
            )

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
