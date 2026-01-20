import time
from collections import deque

import brain.utils.tcp_socket as tcp
import numpy as np
import tensorflow as tf
from brain.multi_trainer import MultiTrainer
from brain.utils.logger import logger

MODEL_SAVE_FREQUENCY_MINUTES = 10


class TrainerSACDiscrete(MultiTrainer):

    def __init__(
        self,
        model_name: str,
        actor: tf.keras.Model,
        critic1: tf.keras.Model,
        critic2: tf.keras.Model,
        actor_optimizer: tf.keras.optimizers.Optimizer,
        critic_optimizer: tf.keras.optimizers.Optimizer,
        nb_env: int,
        num_actions: int,
        fit_step_frequency: int,
        gamma: float,
        alpha: float,
        tau: float,
        target_entropy_scale: float,
        temperature_learning_rate: float,
        batch_size: int,
        grad_norm_clip: float,
        memory_size: int,
    ):

        super().__init__(model_name=model_name, model=actor, optimizer=actor_optimizer, nb_env=nb_env, memory_size=1)

        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2

        self.critic1_target = tf.keras.models.clone_model(critic1)
        self.critic1_target.set_weights(critic1.get_weights())
        self.critic1_target.trainable = False
        self.critic2_target = tf.keras.models.clone_model(critic2)
        self.critic2_target.set_weights(critic2.get_weights())
        self.critic2_target.trainable = False

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.nb_env = nb_env
        self.num_actions = num_actions
        self.fit_step_frequency = fit_step_frequency

        self.gamma = gamma
        #self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size
        self.grad_norm_clip = grad_norm_clip

        self.train_step_count = 0
        self.last_model_save_time = time.time()

        # Learnable temperature for entropy regularization
        self.target_entropy = target_entropy_scale * np.log(num_actions)
        self.log_alpha = tf.Variable(tf.math.log(alpha), dtype=tf.float32, trainable=True)
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=temperature_learning_rate)

        # todo merge with parent
        self.replay = deque(maxlen=memory_size)

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def update_target_model(self, source: tf.keras.Model, target: tf.keras.Model) -> None:
        sw = source.get_weights()
        tw = target.get_weights()
        target.set_weights([self.tau * s + (1.0 - self.tau) * t for s, t in zip(sw, tw)])

    # @tf.function todo
    def train_step(
        self,
        obs: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        dones: tf.Tensor,
        next_obs: tf.Tensor,
    ) -> tuple[float, float, float, float]:
        # Critic target
        next_logits = self.actor(next_obs, training=True)
        next_log_pi = tf.nn.log_softmax(next_logits, axis=-1)
        next_pi = tf.nn.softmax(next_logits, axis=-1)

        q1_next_t = self.critic1_target(next_obs, training=False)
        q2_next_t = self.critic2_target(next_obs, training=False)
        q_next_t = tf.minimum(q1_next_t, q2_next_t)

        v_next = tf.reduce_sum(next_pi * (q_next_t - self.alpha * next_log_pi), axis=-1)

        y = rewards + self.gamma * (1.0 - dones) * v_next

        action_oh = tf.one_hot(actions, self.num_actions)

        with tf.GradientTape(persistent=True) as tape:
            q1 = self.critic1(obs, training=True)
            q2 = self.critic2(obs, training=True)

            q1_a = tf.reduce_sum(q1 * action_oh, axis=-1)  # (B,)
            q2_a = tf.reduce_sum(q2 * action_oh, axis=-1)  # (B,)

            critic1_loss = tf.reduce_mean(tf.square(y - q1_a))
            critic2_loss = tf.reduce_mean(tf.square(y - q2_a))
            critic_loss = critic1_loss + critic2_loss

            # Actor loss
            logits = self.actor(obs, training=True)  # (B, A)
            log_pi = tf.nn.log_softmax(logits, axis=-1)  # (B, A)
            pi = tf.nn.softmax(logits, axis=-1)  # (B, A)

            q1_pi = self.critic1(obs, training=False)  # (B, A)
            q2_pi = self.critic2(obs, training=False)  # (B, A)
            q_pi = tf.minimum(q1_pi, q2_pi)  # (B, A)

            actor_loss = tf.reduce_mean(tf.reduce_sum(pi * (self.alpha * log_pi - q_pi), axis=-1))

            entropy = -tf.reduce_mean(tf.reduce_sum(pi * log_pi, axis=-1))

            alpha_loss = -tf.reduce_mean(
                self.log_alpha * tf.stop_gradient(entropy - self.target_entropy)
            )

        # Gradients critic
        critic_vars = self.critic1.trainable_variables + self.critic2.trainable_variables
        critic_grads = tape.gradient(critic_loss, critic_vars)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.grad_norm_clip)
        self.critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))

        # Gradients actor
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.grad_norm_clip)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Gradients temperature
        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

        del tape

        # Update critic target networks
        self.update_target_model(self.critic1, self.critic1_target)
        self.update_target_model(self.critic2, self.critic2_target)

        return critic_loss, actor_loss, entropy, alpha_loss

    def policy(self, observation: np.ndarray) -> int:
        observation = np.expand_dims(observation, axis=0).astype(np.float32)
        logits = self.actor(tf.convert_to_tensor(observation), training=False)
        action = tf.random.categorical(logits, 1)
        return int(tf.squeeze(action, axis=1).numpy()[0])

    # todo à sortir
    def _sample_batch(self):
        idx = np.random.randint(0, len(self.replay), size=self.batch_size)
        batch = [self.replay[i] for i in idx]
        obs, actions, rewards, dones, next_obs = zip(*batch)

        return (
            tf.convert_to_tensor(np.array(obs, dtype=np.float32)),
            tf.convert_to_tensor(np.array(actions, dtype=np.int32)),
            tf.convert_to_tensor(np.array(rewards, dtype=np.float32)),
            tf.convert_to_tensor(np.array(dones, dtype=np.float32)),
            tf.convert_to_tensor(np.array(next_obs, dtype=np.float32)),
        )

    def fit_model(self) -> None:
        if len(self.replay) < self.batch_size:
            return

        observations, actions, rewards, dones, next_observations = self._sample_batch()
        critic_loss, actor_loss, entropy, alpha_loss = self.train_step(observations, actions, rewards, dones, next_observations)

        with self.tb_writer.as_default():
            tf.summary.scalar("SAC_Discrete/Critic_Loss", critic_loss, step=self.train_step_count)
            tf.summary.scalar("SAC_Discrete/Actor_Loss", actor_loss, step=self.train_step_count)
            tf.summary.scalar("SAC_Discrete/Entropy", entropy, step=self.train_step_count)
            tf.summary.scalar("SAC_Discrete/Alpha_Loss", alpha_loss, step=self.train_step_count)
            tf.summary.scalar("SAC_Discrete/Batch_Reward_Mean", tf.reduce_mean(rewards), step=self.train_step_count)
            tf.summary.scalar("SAC_Discrete/Replay_Buffer_Size", len(self.replay), step=self.train_step_count)

        self.train_step_count += 1

    def run(self) -> None:
        self.accept_connections()
        sockets = self.sockets
        new_simulation_step_count = 0

        while True:
            closed_sockets = []

            for conn, port in sockets:
                obj = tcp.read(conn)

                if obj is not None:
                    message_type = obj["message_type"]

                    if message_type == "policy":
                        observation = np.array(obj["observation"], dtype=np.float32)
                        action = self.policy(observation)
                        tcp.send(conn, {"action": action})

                    elif message_type == "step":
                        observation = np.array(obj["observation"], dtype=np.float32)
                        action = int(obj["action"])
                        reward = float(obj["reward"])
                        done = float(obj["done"])
                        next_observation = np.array(obj["next_observation"], dtype=np.float32)
                        self.replay.append((observation, action, reward, done, next_observation))
                        new_simulation_step_count += 1

                    elif message_type == "terminated":
                        closed_sockets.append(port)
                        logger().info(f"Environment on port {port} a terminé.")

            sockets = [s for s in sockets if s[1] not in closed_sockets]

            if len(sockets) == 0:
                self.save_model()
                break

            if new_simulation_step_count >= self.fit_step_frequency:
                self.fit_model()
                new_simulation_step_count = 0

            if (
                self.train_step_count > 0
                and time.time() - self.last_model_save_time >= MODEL_SAVE_FREQUENCY_MINUTES * 60
            ):
                self.save_model()
                self.last_model_save_time = time.time()
