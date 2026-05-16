import logging

import mlflow
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from corl.memory.prioritized_experience_replay import PrioritizedExperienceReplayBuffer
from corl.memory.transition import Transition as TransitionMemory
from corl.trainer.trainer import Trainer
from corl.utils.config import Config
from corl.model.discrete.actor_critic import ModelActorCritic

logger = logging.getLogger(__name__)


class TrainerTD3(Trainer):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) trainer.

    Implements the TD3 algorithm (Fujimoto et al., 2018) with:

    - **Deterministic actor**: outputs a single action vector squashed to
      ``[-1, 1]`` via ``tanh``. The actor Keras model must output a flat
      vector of shape ``(batch, action_dim)``.
    - **Twin Q-critics**: two independent critic networks; Bellman targets
      use ``min(Q1, Q2)`` to reduce overestimation bias.
    - **Target policy smoothing**: clipped Gaussian noise is added to target
      actions when computing Bellman targets, preventing the policy from
      exploiting narrow peaks in the Q-function.
    - **Delayed policy updates**: the actor and all target networks are
      updated every ``policy_delay`` critic updates (default 2), reducing
      variance in the policy gradient.
    - **Soft target networks**: exponential moving average updates for both
      target critics and the target actor.
    - **Exploration noise**: Gaussian noise added to actions during rollout.
    - **PER replay buffer**: prioritised experience replay with importance-
      sampling correction.

    Attributes:
        gamma: Discount factor.
        tau: Soft-update coefficient for target network updates.
        batch_size: Transitions sampled per gradient step.
        fit_frequency: Steps between gradient updates.
        actor_lr: Learning rate for the actor.
        critic_lr: Learning rate for both critic networks.
        action_dim: Dimensionality of the continuous action space.
        policy_delay: Critic updates between each actor/target update.
        exploration_noise: Std of Gaussian noise added to actions in rollout.
        target_noise: Std of smoothing noise added to target actions.
        target_noise_clip: Absolute clip bound for target smoothing noise.
        max_grad_norm: Optional L2 gradient clipping threshold.
        critic1: Online critic network 1.
        critic2: Online critic network 2.
        target_critic1: Soft-updated copy of ``critic1``.
        target_critic2: Soft-updated copy of ``critic2``.
        target_actor: Soft-updated copy of the actor.
        experience_replay: PER buffer.
        per_beta: Current IS correction exponent.
        per_beta_increment: Per-transition increment for ``per_beta``.
    """

    gamma: float
    tau: float
    batch_size: int
    fit_frequency: int
    actor_lr: float
    critic_lr: float
    action_dim: int
    policy_delay: int
    exploration_noise: float
    target_noise: float
    target_noise_clip: float
    max_grad_norm: float | None
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
        policy_delay: int = 2,
        exploration_noise: float = 0.1,
        target_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        per_size: int = 100_000,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        max_grad_norm: float | None = None,
    ):
        """
        Initialise the TD3 trainer.

        The actor Keras model must output a flat vector of shape
        ``(batch, action_ the deterministic action. A ``tanh``dim)``
        output activation is recommended so actions are already in
        ``[-1, 1]``. Both critic Keras models must accept a concatenated
        ``(obs, action)`` input and output a scalar Q-value.

        Args:
            config: Application configuration.
            actor: Keras model for the deterministic actor (outputs action).
            critic1: Keras model for Q-network 1.
            critic2: Keras model for Q-network 2.
            action_dim: Dimensionality of the continuous action space.
            model_checkpoint_frequency: Transitions between checkpoint saves.
            gamma: Discount factor in ``[0, 1]``.
            tau: Soft-update coefficient. ``0.005`` is the TD3 default.
            batch_size: Number of transitions per gradient step.
            fit_frequency: Steps between gradient updates.
            actor_lr: Learning rate for the actor Adam optimiser.
            critic_lr: Learning rate for both critic Adam optimisers.
            policy_delay: Number of critic updates per actor update.
                The TD3 paper recommends ``2``.
            exploration_noise: Std of Gaussian noise added to actions
                during environment interaction (rollout exploration).
            target_noise: Std of smoothing noise added to target actions
                when computing Bellman targets.
            target_noise_clip: Absolute clip bound for ``target_noise``.
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
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.max_grad_norm = max_grad_norm
        self.per_beta = per_beta_start
        self._critic_update_count = 0

        # Twin critics and their soft-updated targets
        self.critic1 = critic1
        self.critic2 = critic2
        self.target_critic1 = tf.keras.models.clone_model(critic1)
        self.target_critic2 = tf.keras.models.clone_model(critic2)
        self.target_critic1.set_weights(critic1.get_weights())
        self.target_critic2.set_weights(critic2.get_weights())

        # Target actor (TD3 requires a target for the actor too)
        self.target_actor = tf.keras.models.clone_model(actor)
        self.target_actor.set_weights(actor.get_weights())

        # Optimisers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        self.transition = TransitionMemory()
        self.experience_replay = PrioritizedExperienceReplayBuffer(
            capacity=per_size, alpha=per_alpha
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
            "action_dim": self.action_dim,
            "policy_delay": self.policy_delay,
            "exploration_noise": self.exploration_noise,
            "target_noise": self.target_noise,
            "target_noise_clip": self.target_noise_clip,
            "per_alpha": self.experience_replay.alpha,
            "per_beta_start": self.per_beta,
            "max_grad_norm": self.max_grad_norm
            if self.max_grad_norm is not None
            else "disabled",
        }

    def policy(self, observations: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Return noisy deterministic actions for a batch of observations.

        The actor output is a deterministic action in ``[-1, 1]``.
        Gaussian exploration noise (``exploration_noise``) is added and
        the result is clipped back to ``[-1, 1]``.

        Args:
            observations: Batch of observations, shape ``(batch_size, obs_dim)``.

        Returns:
            NDArray[np.float32]: Actions with exploration noise,
            shape ``(batch_size, action_dim)``.
        """
        obs_t = tf.constant(observations, dtype=tf.float32)
        actions = self.model.actor(obs_t, training=False).numpy()
        noise = np.random.normal(0.0, self.exploration_noise, actions.shape)
        return np.clip(actions + noise, -1.0, 1.0).astype(np.float32)

    def fit_model(self) -> dict[str, float] | None:
        """
        Sample a batch from PER and perform one TD3 gradient update.

        Returns ``None`` if the buffer is not large enough. Otherwise:

        1. Samples a stratified batch from PER with the current ``per_beta``.
        2. Computes smoothed target actions using the target actor and clipped
           Gaussian noise.
        3. Computes Bellman targets using ``min(Q1_target, Q2_target)``.
        4. Updates both critics (MSE on TD targets).
        5. Every ``policy_delay`` critic updates:

           a. Updates the actor by maximising ``E[Q1(s, actor(s))]``.
           b. Soft-updates both target critics and the target actor.

        6. Updates PER priorities from the mean TD-error of the two critics.

        Returns:
            dict with keys ``"critic1_loss"``, ``"critic2_loss"``,
            ``"mean_reward"``, ``"mean_q1"``, and (every ``policy_delay``
            updates) ``"actor_loss"``.
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

        # ---- Target policy smoothing ----------------------------------
        next_actions = self.target_actor(next_obs_t, training=False)
        noise = tf.clip_by_value(
            tf.random.normal(tf.shape(next_actions), stddev=self.target_noise),
            -self.target_noise_clip,
            self.target_noise_clip,
        )
        next_actions = tf.clip_by_value(next_actions + noise, -1.0, 1.0)

        # ---- Bellman targets (no gradient) ----------------------------
        next_input = tf.concat([next_obs_t, next_actions], axis=-1)
        next_q1 = tf.squeeze(self.target_critic1(next_input, training=False), axis=-1)
        next_q2 = tf.squeeze(self.target_critic2(next_input, training=False), axis=-1)
        td_targets = rew_t + self.gamma * non_terminal * tf.minimum(next_q1, next_q2)

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

        self._critic_update_count += 1

        # ---- PER priority update --------------------------------------
        td_errors = (td_errors1.numpy() + td_errors2.numpy()) / 2.0
        self.experience_replay.update_priorities(idxs, td_errors)

        metrics: dict[str, float] = {
            "critic1_loss": float(c1_loss.numpy()),
            "critic2_loss": float(c2_loss.numpy()),
            "mean_reward": float(np.mean(rewards)),
            "mean_q1": float(tf.reduce_mean(q1).numpy()),
        }

        # ---- Delayed actor + target updates ---------------------------
        if self._critic_update_count % self.policy_delay == 0:
            with tf.GradientTape() as actor_tape:
                new_actions = self.model.actor(obs_t, training=True)
                actor_input = tf.concat([obs_t, new_actions], axis=-1)
                actor_loss = -tf.reduce_mean(
                    tf.squeeze(self.critic1(actor_input, training=False), axis=-1)
                )

            self._apply_gradients(
                actor_tape,
                actor_loss,
                self.model.actor.trainable_variables,
                self.actor_optimizer,
            )

            self._soft_update(self.model.actor, self.target_actor)
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)

            metrics["actor_loss"] = float(actor_loss.numpy())

        return metrics

    def run(self, epochs: int) -> None:
        """
        Execute the TD3 training loop for ``epochs`` transitions.

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

        logger.info(f"Starting TD3 training for {epochs} transitions")

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
