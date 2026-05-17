import logging

import mlflow
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from corl.memory.prioritized_experience_replay import PrioritizedExperienceReplayBuffer
from corl.memory.transition import Transition as TransitionMemory
from corl.model.discrete.actor_critic import ModelActorCritic
from corl.trainer.trainer import Trainer
from corl.utils.config import Config

logger = logging.getLogger(__name__)

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class TrainerSAC(Trainer):
    """
    Soft Actor-Critic (SAC) trainer for continuous action spaces.

    Implements the SAC algorithm with:

    - **Stochastic Gaussian actor**: samples actions via the reparameterisation
      trick, squashed through ``tanh`` to ``[-1, 1]``.
    - **Twin Q-critics**: two independent critic networks; Bellman targets use
      ``min(Q1, Q2)`` to reduce overestimation bias.
    - **Soft target critics**: exponential moving average updates
      (``τ * online + (1-τ) * target``) replace hard copies.
    - **Automatic entropy tuning**: the temperature ``alpha`` is learnt to
      match a fixed ``target_entropy`` (typically ``-action_dim``); set
      ``auto_alpha=False`` to use a fixed value instead.
    - **PER replay buffer**: prioritised experience replay with importance-
      sampling correction.

    Attributes:
        gamma: Discount factor.
        tau: Soft-update coefficient for target critic updates.
        batch_size: Transitions sampled per gradient step.
        fit_frequency: Steps between gradient updates.
        actor_lr: Learning rate for the actor.
        critic_lr: Learning rate for both critic networks.
        alpha: Current entropy temperature (scalar).
        auto_alpha: Whether ``alpha`` is adapted automatically.
        target_entropy: Desired entropy level (used when ``auto_alpha=True``).
        log_alpha: Learnable log-temperature variable (``auto_alpha=True`` only).
        alpha_optimizer: Optimiser for ``log_alpha``.
        actor_optimizer: Adam optimiser for the actor network.
        critic1_optimizer: Adam optimiser for critic 1.
        critic2_optimizer: Adam optimiser for critic 2.
        critic1: Online critic network 1 — takes ``(obs, action)`` → Q-value.
        critic2: Online critic network 2 — takes ``(obs, action)`` → Q-value.
        target_critic1: Soft-updated copy of ``critic1``.
        target_critic2: Soft-updated copy of ``critic2``.
        action_dim: Dimensionality of the continuous action space.
        experience_replay: PER buffer.
        transition: Helper for assembling step results into TD transitions.
        per_beta: Current IS correction exponent.
        per_beta_increment: Per-transition increment for ``per_beta``.
    """

    gamma: float
    tau: float
    batch_size: int
    fit_frequency: int
    actor_lr: float
    critic_lr: float
    alpha: float
    auto_alpha: bool
    target_entropy: float
    action_dim: int
    per_beta: float
    per_beta_increment: float

    def __init__(
        self,
        config: Config,
        actor: tf.keras.Model,
        critic1: tf.keras.Model,
        critic2: tf.keras.Model,
        action_dim: int,
        model_checkpoint_frequency: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        fit_frequency: int = 1,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: float | None = None,
        per_size: int = 100_000,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        max_grad_norm: float | None = None,
    ):
        """
        Initialise the SAC trainer.

        The actor Keras model must output a flat vector of shape
        ``(batch, action_dim * 2)`` where the first ``action_dim`` values are
        the mean and the last ``action_dim`` values are the log-standard-deviation
        of the Gaussian policy. Both critic Keras models must accept a
        concatenated ``(obs, action)`` input and output a scalar Q-value.

        Args:
            config: Application configuration.
            actor: Keras model for the Gaussian actor (outputs mean + log_std).
            critic1: Keras model for Q-network 1.
            critic2: Keras model for Q-network 2.
            action_dim: Dimensionality of the continuous action space.
            model_checkpoint_frequency: Transitions between checkpoint saves.
            gamma: Discount factor in ``[0, 1]``.
            tau: Soft-update coefficient. ``1.0`` = hard copy; ``0.005``
                is a common default.
            batch_size: Number of transitions per gradient step.
            fit_frequency: Steps between gradient updates.
            actor_lr: Learning rate for the actor Adam optimiser.
            critic_lr: Learning rate for both critic Adam optimisers.
            alpha: Initial entropy temperature. Ignored when
                ``auto_alpha=True`` after the first update.
            auto_alpha: If ``True``, adapt ``alpha`` automatically to
                match ``target_entropy``.
            target_entropy: Desired policy entropy. Defaults to
                ``-action_dim`` when ``None``.
            per_size: Capacity of the replay buffer.
            per_alpha: PER priority exponent.
            per_beta_start: Initial IS correction exponent, annealed to
                ``1.0`` over training.
            max_grad_norm: L2 norm cap for gradient clipping. ``None``
                disables clipping.
        """

        model_stub = ModelActorCritic(
            actor=actor, critic=critic1, action_size=action_dim
        )

        super().__init__(
            model=model_stub,
            config=config,
            model_checkpoint_frequency=model_checkpoint_frequency,
        )

        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.fit_frequency = fit_frequency
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.target_entropy = (
            float(-action_dim) if target_entropy is None else target_entropy
        )
        self.max_grad_norm = max_grad_norm
        self.per_beta = per_beta_start

        # Twin critics and their soft-updated targets
        self.critic1 = critic1
        self.critic2 = critic2
        self.target_critic1 = tf.keras.models.clone_model(critic1)
        self.target_critic2 = tf.keras.models.clone_model(critic2)
        self.target_critic1.set_weights(critic1.get_weights())
        self.target_critic2.set_weights(critic2.get_weights())

        # Optimisers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        # Automatic entropy tuning
        if auto_alpha:
            self.log_alpha = tf.Variable(
                tf.math.log(tf.constant(alpha)), trainable=True, dtype=tf.float32
            )
            self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None

        self.transition = TransitionMemory()
        self.experience_replay = PrioritizedExperienceReplayBuffer(
            capacity=per_size, alpha=per_alpha
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_action(self, observations: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Sample actions from the squashed Gaussian policy.

        Runs the actor forward pass to obtain ``(mean, log_std)``, samples
        using the reparameterisation trick, and squashes through ``tanh``.
        The log-probability is corrected for the ``tanh`` squashing.

        Args:
            observations: Float tensor of shape ``(batch, obs_dim)``.

        Returns:
            Tuple ``(actions, log_probs)`` both of shape ``(batch, action_dim)``.
        """
        output = self.model.actor(observations, training=True)
        mean, log_std = tf.split(output, 2, axis=-1)
        mean = tf.clip_by_value(mean, -4.0, 4.0)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)

        eps = tf.random.normal(tf.shape(mean))
        raw = mean + std * eps  # reparameterisation

        actions = tf.tanh(raw)
        actions = tf.where(tf.math.is_finite(actions), actions, tf.zeros_like(actions))

        # Log-prob with tanh squashing correction
        log_probs = (
            -0.5 * tf.square(eps)
            - log_std
            - 0.5 * tf.math.log(2.0 * np.pi)
            - tf.math.log(1.0 - tf.square(actions) + 1e-6)
        )
        log_probs = tf.reduce_sum(log_probs, axis=-1, keepdims=True)

        return actions, log_probs

    def _apply_gradients(
        self,
        tape: tf.GradientTape,
        loss: tf.Tensor,
        variables: list[tf.Variable],
        optimizer: tf.keras.optimizers.Optimizer,
    ) -> None:
        grads = tape.gradient(loss, variables)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        optimizer.apply_gradients(zip(grads, variables))

    def _soft_update(self, online: tf.keras.Model, target: tf.keras.Model) -> None:
        """Polyak-average ``online`` weights into ``target``."""
        for w_o, w_t in zip(online.weights, target.weights):
            w_t.assign(self.tau * w_o + (1.0 - self.tau) * w_t)

    # ------------------------------------------------------------------
    # Trainer interface
    # ------------------------------------------------------------------

    def params(self) -> dict[str, str | int | float]:
        return {
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "fit_frequency": self.fit_frequency,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "alpha_initial": self.alpha,
            "auto_alpha": self.auto_alpha,
            "target_entropy": self.target_entropy,
            "action_dim": self.action_dim,
            "per_alpha": self.experience_replay.alpha,
            "per_beta_start": self.per_beta,
            "max_grad_norm": self.max_grad_norm
            if self.max_grad_norm is not None
            else "disabled",
        }

    def policy(self, observations: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Sample continuous actions for a batch of observations.

        Actions are sampled from the squashed Gaussian and returned as a
        float32 array of shape ``(batch_size, action_dim)``.

        Args:
            observations: Batch of observations, shape ``(batch_size, obs_dim)``.

        Returns:
            NDArray[np.float32]: Sampled actions, shape ``(batch_size, action_dim)``.
        """
        obs_t = tf.constant(observations, dtype=tf.float32)
        actions, _ = self._sample_action(obs_t)
        return actions.numpy().astype(np.float32)

    def fit_model(self) -> dict[str, float] | None:
        """
        Sample a batch from PER and perform one SAC gradient update.

        Returns ``None`` if the buffer is not large enough. Otherwise:

        1. Samples a stratified batch from PER with the current ``per_beta``.
        2. Computes soft Bellman targets using the target critics and the
           current actor entropy bonus.
        3. Updates both critics (MSE on TD targets).
        4. Updates the actor by maximising ``E[Q - α log π]``.
        5. If ``auto_alpha=True``, updates the temperature to track
           ``target_entropy``.
        6. Soft-updates both target critics.
        7. Updates PER priorities from the mean TD-error of the two critics.

        Returns:
            dict with keys ``"critic1_loss"``, ``"critic2_loss"``,
            ``"actor_loss"``, ``"alpha"``, ``"mean_reward"``,
            ``"mean_q1"``, and optionally ``"alpha_loss"``.
        """
        if len(self.experience_replay) < self.batch_size:
            return None

        observations, actions, rewards, next_observations, terminals, idxs, weights = (
            self.experience_replay.sample(self.batch_size, self.per_beta)
        )

        obs_t = tf.constant(observations, dtype=tf.float32)
        act_t = tf.constant(actions, dtype=tf.float32)
        rew_t = tf.constant(rewards, dtype=tf.float32)
        next_obs_t = tf.constant(next_observations, dtype=tf.float32)
        non_terminal = tf.constant(~terminals, dtype=tf.float32)
        weights_t = tf.constant(weights, dtype=tf.float32)

        # ---- Bellman targets (no gradient) ----------------------------
        next_actions, next_log_probs = self._sample_action(next_obs_t)
        next_input = tf.concat([next_obs_t, next_actions], axis=-1)
        next_q1 = tf.squeeze(self.target_critic1(next_input, training=False), axis=-1)
        next_q2 = tf.squeeze(self.target_critic2(next_input, training=False), axis=-1)
        next_q = tf.minimum(next_q1, next_q2) - self.alpha * tf.squeeze(
            next_log_probs, axis=-1
        )
        td_targets = rew_t + self.gamma * non_terminal * next_q

        # ---- Critics --------------------------------------------------
        critic_input = tf.concat([obs_t, act_t], axis=-1)

        with tf.GradientTape() as tape1:
            q1 = tf.squeeze(self.critic1(critic_input, training=True), axis=-1)
            td_errors1 = td_targets - q1
            c1_loss = tf.reduce_mean(weights_t * tf.square(td_errors1))

        self._apply_gradients(
            tape1, c1_loss, self.critic1.trainable_variables, self.critic1_optimizer
        )

        with tf.GradientTape() as tape2:
            q2 = tf.squeeze(self.critic2(critic_input, training=True), axis=-1)
            td_errors2 = td_targets - q2
            c2_loss = tf.reduce_mean(weights_t * tf.square(td_errors2))

        self._apply_gradients(
            tape2, c2_loss, self.critic2.trainable_variables, self.critic2_optimizer
        )

        # ---- Actor ----------------------------------------------------
        with tf.GradientTape() as actor_tape:
            new_actions, log_probs = self._sample_action(obs_t)
            new_input = tf.concat([obs_t, new_actions], axis=-1)
            q1_new = tf.squeeze(self.critic1(new_input, training=False), axis=-1)
            q2_new = tf.squeeze(self.critic2(new_input, training=False), axis=-1)
            min_q_new = tf.minimum(q1_new, q2_new)
            actor_loss = tf.reduce_mean(
                self.alpha * tf.squeeze(log_probs, axis=-1) - min_q_new
            )

        self._apply_gradients(
            actor_tape,
            actor_loss,
            self.model.actor.trainable_variables,
            self.actor_optimizer,
        )

        # ---- Alpha (entropy temperature) ------------------------------
        alpha_loss = None
        if self.auto_alpha and self.log_alpha is not None:
            with tf.GradientTape() as alpha_tape:
                detached_log_probs = tf.stop_gradient(log_probs)
                alpha_loss = -tf.reduce_mean(
                    self.log_alpha * (detached_log_probs + self.target_entropy)
                )
            self._apply_gradients(
                alpha_tape,
                alpha_loss,
                [self.log_alpha],
                self.alpha_optimizer,
            )
            self.alpha = float(tf.exp(self.log_alpha).numpy())

        # ---- Soft target update ---------------------------------------
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        # ---- PER priority update --------------------------------------
        td_errors = (td_errors1.numpy() + td_errors2.numpy()) / 2.0
        self.experience_replay.update_priorities(idxs, td_errors)

        metrics = {
            "critic1_loss": float(c1_loss.numpy()),
            "critic2_loss": float(c2_loss.numpy()),
            "actor_loss": float(actor_loss.numpy()),
            "alpha": self.alpha,
            "mean_reward": float(np.mean(rewards)),
            "mean_q1": float(tf.reduce_mean(q1).numpy()),
        }
        if alpha_loss is not None:
            metrics["alpha_loss"] = float(alpha_loss.numpy())

        return metrics

    def run(self, epochs: int) -> None:
        """
        Execute the SAC training loop for ``epochs`` transitions.

        Logs hyperparameters to MLflow, then repeatedly calls
        :meth:`~corl.trainer.trainer.Trainer.training_step` to collect
        environment interactions. After each collected batch:

        - Transitions are pushed into the PER buffer.
        - :meth:`fit_model` is called every ``fit_frequency`` steps;
          metrics are logged to MLflow when a fit occurs.
        - A model checkpoint is saved every
          ``model_checkpoint_frequency`` steps.
        - ``per_beta`` is annealed toward ``1.0`` each transition.

        On completion, the final model is saved and the trainer is closed.

        Args:
            epochs: Total number of simulation steps to run.

        Raises:
            ValueError: If ``epochs`` is less than ``1``.
        """
        if epochs < 1:
            raise ValueError(f"Number of epochs must be >= 1, got {epochs}")

        mlflow.log_params(self.params())

        training_step_count = 0
        episode_count = 0
        last_fit = 0
        last_checkpoint = 0
        self.per_beta_increment = (1.0 - self.per_beta) / epochs

        logger.info(f"Starting SAC training for {epochs} transitions")

        while training_step_count < epochs:
            steps = self.training_step()

            transitions = [
                transition
                for step_key, step_result in steps
                for transition in self.transition.make(step_key.worker_id, step_result)
            ]

            for transition in transitions:
                training_step_count += 1

                if transition.current_step.done:
                    episode_count += 1

                # Actions are stored as list[float]; convert to numpy for PER
                action = np.array(transition.current_step.action, dtype=np.float32)
                self.experience_replay.add(
                    transition.current_step.observation,
                    action,
                    transition.current_step.reward,
                    transition.next_step.observation
                    if transition.next_step
                    else transition.current_step.observation,
                    transition.current_step.done,
                )

                self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

            if training_step_count - last_fit >= self.fit_frequency:
                fit_metrics = self.fit_model()
                if fit_metrics:
                    mlflow.log_metrics(
                        fit_metrics
                        | {
                            "episode": episode_count,
                            "transition_per_episode": round(
                                training_step_count / episode_count, 2
                            )
                            if episode_count > 0
                            else 0.0,
                            "per_size": len(self.experience_replay),
                            "per_beta": self.per_beta,
                        },
                        step=training_step_count,
                    )
                last_fit = training_step_count

            if training_step_count - last_checkpoint >= self.model_checkpoint_frequency:
                self.model.save_weights(self.model_dir, checkpoint=True)
                last_checkpoint = training_step_count

            logger.debug(
                f"Processed {len(steps)} steps with {len(transitions)} "
                f"transitions. {training_step_count}/{epochs}."
            )

        self.model.save(self.model_dir)
        self.close()
