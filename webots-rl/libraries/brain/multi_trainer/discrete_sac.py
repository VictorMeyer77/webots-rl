python
import json
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
        critic1_target: tf.keras.Model,
        critic2_target: tf.keras.Model,
        actor_optimizer: tf.keras.optimizers.Optimizer,
        critic_optimizer: tf.keras.optimizers.Optimizer,
        nb_env: int,
        num_actions: int,
        fit_step_frequency: int,
        gamma: float,
        alpha: float,
        tau: float,
        batch_size: int,
        grad_norm_clip: float,
        memory_size: int = 200_000,
    ):
        # MultiTrainer est conservé pour la gestion sockets, TB, save_model, etc.
        # On passe un "model" dummy (actor) pour rester compatible avec l'API existante.
        super().__init__(model_name=model_name, model=actor, optimizer=actor_optimizer, nb_env=nb_env, memory_size=1)

        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.critic1_target = critic1_target
        self.critic2_target = critic2_target

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.nb_env = nb_env
        self.num_actions = num_actions
        self.fit_step_frequency = fit_step_frequency

        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size
        self.grad_norm_clip = grad_norm_clip

        self.step_count = 0
        self.last_model_save_time = time.time()

        # Replay buffer (global)
        self.replay = deque(maxlen=memory_size)

        # Dernière obs par env (port) pour reconstruire (s, a, r, done, s2)
        self._last_obs_by_port: dict[int, np.ndarray] = {}

    def _soft_update(self, source: tf.keras.Model, target: tf.keras.Model) -> None:
        sw = source.get_weights()
        tw = target.get_weights()
        target.set_weights([self.tau * s + (1.0 - self.tau) * t for s, t in zip(sw, tw)])

    @tf.function
    def _train_step(
        self,
        obs: tf.Tensor,        # (B, H, W, C)
        actions: tf.Tensor,    # (B,)
        rewards: tf.Tensor,    # (B,)
        dones: tf.Tensor,      # (B,)
        next_obs: tf.Tensor,   # (B, H, W, C)
    ):
        # -------- Critic target (discret) --------
        next_logits = self.actor(next_obs, training=True)  # (B, A)
        next_log_pi = tf.nn.log_softmax(next_logits, axis=-1)  # (B, A)
        next_pi = tf.nn.softmax(next_logits, axis=-1)  # (B, A)

        q1_next_t = self.critic1_target(next_obs, training=False)  # (B, A)
        q2_next_t = self.critic2_target(next_obs, training=False)  # (B, A)
        q_next_t = tf.minimum(q1_next_t, q2_next_t)  # (B, A)

        # V(s') = Σ_a π(a|s') [ Q(s',a) - alpha * log π(a|s') ]
        v_next = tf.reduce_sum(next_pi * (q_next_t - self.alpha * next_log_pi), axis=-1)  # (B,)

        y = rewards + self.gamma * (1.0 - dones) * v_next  # (B,)

        action_oh = tf.one_hot(actions, self.num_actions)  # (B, A)

        with tf.GradientTape(persistent=True) as tape:
            q1 = self.critic1(obs, training=True)  # (B, A)
            q2 = self.critic2(obs, training=True)  # (B, A)

            q1_a = tf.reduce_sum(q1 * action_oh, axis=-1)  # (B,)
            q2_a = tf.reduce_sum(q2 * action_oh, axis=-1)  # (B,)

            critic1_loss = tf.reduce_mean(tf.square(y - q1_a))
            critic2_loss = tf.reduce_mean(tf.square(y - q2_a))
            critic_loss = critic1_loss + critic2_loss

            # -------- Actor loss (discret) --------
            logits = self.actor(obs, training=True)  # (B, A)
            log_pi = tf.nn.log_softmax(logits, axis=-1)  # (B, A)
            pi = tf.nn.softmax(logits, axis=-1)  # (B, A)

            q1_pi = self.critic1(obs, training=False)  # (B, A)
            q2_pi = self.critic2(obs, training=False)  # (B, A)
            q_pi = tf.minimum(q1_pi, q2_pi)  # (B, A)

            # J_pi = E_s [ Σ_a π(a|s) (alpha * log π(a|s) - Q(s,a)) ]
            actor_loss = tf.reduce_mean(tf.reduce_sum(pi * (self.alpha * log_pi - q_pi), axis=-1))

            entropy = -tf.reduce_mean(tf.reduce_sum(pi * log_pi, axis=-1))

        # Optim critic
        critic_vars = self.critic1.trainable_variables + self.critic2.trainable_variables
        critic_grads = tape.gradient(critic_loss, critic_vars)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.grad_norm_clip)
        self.critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))

        # Optim actor
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.grad_norm_clip)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        del tape

        # Soft update targets
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return critic_loss, actor_loss, entropy

    def policy(self, observation: np.ndarray) -> int:
        obs = np.expand_dims(observation, axis=0).astype(np.float32)
        logits = self.actor(tf.convert_to_tensor(obs), training=False)
        action = tf.random.categorical(logits, 1)
        return int(tf.squeeze(action, axis=1).numpy()[0])

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

        obs, actions, rewards, dones, next_obs = self._sample_batch()
        critic_loss, actor_loss, entropy = self._train_step(obs, actions, rewards, dones, next_obs)

        with self.tb_writer.as_default():
            tf.summary.scalar("SAC_Discrete/Critic_Loss", critic_loss, step=self.step_count)
            tf.summary.scalar("SAC_Discrete/Actor_Loss", actor_loss, step=self.step_count)
            tf.summary.scalar("SAC_Discrete/Entropy", entropy, step=self.step_count)

        self.step_count += 1

    def run(self) -> None:
        self.accept_connections()
        sockets = self.sockets

        while True:
            closed_sockets = []

            for conn, port in sockets:
                msg = tcp.read(conn)
                if msg is None:
                    continue

                obj = json.loads(msg)
                message_type = obj["message_type"]

                if message_type == "policy":
                    observation = np.array(obj["observation"], dtype=np.float32)
                    action = self.policy(observation)
                    # On renvoie seulement l'action (SAC n'a pas besoin de "value" ici côté env)
                    tcp.send(conn, json.dumps({"action": action}))

                elif message_type == "step":
                    # On reconstruit la transition à partir de l'obs précédente stockée par port
                    obs = np.array(obj["observation"], dtype=np.float32)
                    action = int(obj["action"])
                    reward = float(obj["reward"])
                    done = float(obj["done"])

                    if port in self._last_obs_by_port:
                        prev_obs = self._last_obs_by_port[port]
                        self.replay.append((prev_obs, action, reward, done, obs))

                    self._last_obs_by_port[port] = obs

                elif message_type == "terminated":
                    closed_sockets.append(port)
                    self._last_obs_by_port.pop(port, None)
                    logger().info(f"Environment on port {port} a terminé.")

            sockets = [s for s in sockets if s[1] not in closed_sockets]

            if len(sockets) == 0:
                self.save_model()
                break

            # Entraînement périodique
            if self.step_count == 0 or (self.step_count % self.fit_step_frequency == 0):
                self.fit_model()

            if self.step_count > 0 and time.time() - self.last_model_save_time >= MODEL_SAVE_FREQUENCY_MINUTES * 60:
                self.save_model()
                self.last_model_save_time = time.time()
