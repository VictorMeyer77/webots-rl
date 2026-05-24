import json
import logging
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from corl.memory.transition import Transition as TransitionMemory
from corl.model.discrete.actor_critic import ModelActorCritic
from corl.schemas.tracker import Transition as TransitionSchema
from corl.trainer.trainer import Trainer
from corl.utils.config import Config

logger = logging.getLogger(__name__)

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class TrainerPPOContinuous(Trainer):
    """
    Proximal Policy Optimisation (PPO) trainer for continuous action spaces.

    Implements PPO-Clip with a squashed Gaussian actor (mean + log_std output,
    sampled via the reparameterisation trick and squashed through ``tanh`` to
    ``[-1, 1]``). The critic is a separate network that estimates state values.
    On-policy data is reused for multiple mini-batch gradient steps, bounded
    by the clipped probability-ratio objective.

    Key design decisions:

    - **Squashed Gaussian actor**: outputs ``action_dim * 2`` values
      ``[mean, log_std]``; actions are sampled as ``tanh(mean + std * ε)``.
    - **Tanh log-prob correction**: log-probabilities include the Jacobian
      correction ``log(1 - tanh²(raw) + ε)`` to account for the squashing.
    - **atanh recovery**: stored squashed actions are inverted via ``atanh``
      in :meth:`update` to recompute log-probabilities under the new policy
      without requiring an extra storage field.
    - **Clipped surrogate objective**: the probability ratio is clipped to
      ``[1 - clip_range, 1 + clip_range]``.
    - **Advantage normalisation**: advantages are standardised per epoch.
    - **On-policy**: the rollout buffer is discarded after each update cycle.

    Attributes:
        action_dim: Dimensionality of the continuous action space.
        gamma: Discount factor for future rewards.
        clip_range: Clipping parameter ε for the surrogate objective.
        ppo_epochs: Number of optimisation passes over each collected rollout.
        mini_batch_size: Number of transitions per mini-batch.
        entropy_coeff: Weight of the differential entropy bonus.
        value_loss_coeff: Scaling factor for the critic MSE loss.
        actor_lr: Learning rate for the actor optimiser.
        critic_lr: Learning rate for the critic optimiser.
        update_frequency: Transitions collected between parameter updates.
        max_grad_norm: Maximum L2 norm for gradient clipping. ``None``
            disables clipping.
        actor_optimizer: Adam optimiser for the actor network.
        critic_optimizer: Adam optimiser for the critic network.
        transition: Helper for assembling step results into TD transitions.
    """

    action_dim: int
    gamma: float
    clip_range: float
    ppo_epochs: int
    mini_batch_size: int
    entropy_coeff: float
    value_loss_coeff: float
    actor_lr: float
    critic_lr: float
    update_frequency: int
    max_grad_norm: float | None
    actor_optimizer: tf.keras.optimizers.Adam
    critic_optimizer: tf.keras.optimizers.Adam
    transition: TransitionMemory

    def __init__(
        self,
        config: Config,
        actor: tf.keras.Model,
        critic: tf.keras.Model,
        action_dim: int,
        checkpoint_frequency: int,
        gamma: float = 0.99,
        clip_range: float = 0.2,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        entropy_coeff: float = 0.0,
        value_loss_coeff: float = 0.5,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        update_frequency: int = 2048,
        max_grad_norm: float | None = 0.5,
        checkpoint_id: str | None = None,
    ):
        """
        Initialise the continuous PPO trainer.

        The actor Keras model must output a flat vector of shape
        ``(batch, action_dim * 2)`` where the first ``action_dim`` values are
        the mean and the last ``action_dim`` values are the log-standard-deviation
        of the Gaussian policy. The critic must accept an observation and output
        a scalar state-value estimate.

        Args:
            config: Application configuration.
            actor: Keras model for the Gaussian actor (outputs mean + log_std).
            critic: Keras model for the state-value function.
            action_dim: Dimensionality of the continuous action space.
            checkpoint_frequency: Transitions between checkpoint saves.
            gamma: Discount factor in ``[0, 1]``.
            clip_range: Clipping parameter ε for the surrogate objective.
                Typical values are ``0.1``–``0.3``.
            ppo_epochs: Number of passes over the rollout buffer per update.
            mini_batch_size: Number of transitions per mini-batch within
                each PPO epoch.
            entropy_coeff: Weight of the differential entropy bonus. Higher
                values encourage more exploration. Often set to ``0`` for
                continuous tasks where the Gaussian already explores.
            value_loss_coeff: Scaling factor for the critic MSE loss.
            actor_lr: Learning rate for the actor's Adam optimiser.
            critic_lr: Learning rate for the critic's Adam optimiser.
            update_frequency: Number of transitions to collect before each
                parameter update.
            max_grad_norm: Maximum L2 norm for gradient clipping. Set to
                ``None`` to disable.
            checkpoint_id: If provided, resume training from this checkpoint
                via :meth:`recovery`. When ``None``, a fresh training session
                is started. Defaults to ``None``.
        """
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_range = clip_range
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.update_frequency = update_frequency
        self.max_grad_norm = max_grad_norm

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.transition = TransitionMemory()

        model_stub = ModelActorCritic(
            actor=actor, critic=critic, action_size=action_dim
        )
        super().__init__(
            model=model_stub,
            config=config,
            checkpoint_frequency=checkpoint_frequency,
            checkpoint_id=checkpoint_id,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gaussian_log_prob(
        self,
        raw: tf.Tensor,
        mean: tf.Tensor,
        log_std: tf.Tensor,
    ) -> tf.Tensor:
        """
        Log-probability of ``raw`` under a diagonal Gaussian with tanh correction.

        Computes ``log N(raw | mean, exp(log_std)²)`` summed over action
        dimensions, then subtracts ``log(1 - tanh(raw)² + ε)`` to correct
        for the tanh squashing transform.

        Args:
            raw: Pre-tanh action tensor, shape ``(batch, action_dim)``.
            mean: Gaussian mean tensor, shape ``(batch, action_dim)``.
            log_std: Log standard deviation, shape ``(batch, action_dim)``.

        Returns:
            Log-probability tensor of shape ``(batch,)``.
        """
        std = tf.exp(log_std)
        log_prob = (
            -0.5 * tf.square((raw - mean) / (std + 1e-8))
            - log_std
            - 0.5 * tf.math.log(2.0 * np.pi)
        )
        tanh_correction = tf.math.log(1.0 - tf.square(tf.tanh(raw)) + 1e-6)
        return tf.reduce_sum(log_prob - tanh_correction, axis=-1)

    def _sample_action(self, observations: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Sample a squashed Gaussian action for each observation.

        Runs the actor forward pass to obtain ``(mean, log_std)``, samples
        using the reparameterisation trick, and squashes through ``tanh``.

        Args:
            observations: Float tensor of shape ``(batch, obs_dim)``.

        Returns:
            Tuple ``(actions, log_probs)`` where ``actions`` has shape
            ``(batch, action_dim)`` with values in ``(-1, 1)`` and
            ``log_probs`` has shape ``(batch,)``.
        """
        output = self.model.actor(observations, training=True)
        mean, log_std = tf.split(output, 2, axis=-1)
        mean = tf.clip_by_value(mean, -4.0, 4.0)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        eps = tf.random.normal(tf.shape(mean))
        raw = mean + tf.exp(log_std) * eps
        actions = tf.tanh(raw)
        actions = tf.where(tf.math.is_finite(actions), actions, tf.zeros_like(actions))
        log_probs = self._gaussian_log_prob(raw, mean, log_std)
        return actions, log_probs

    def _apply_gradients(
        self,
        tape: tf.GradientTape,
        loss: tf.Tensor,
        variables: list[tf.Variable],
        optimizer: tf.keras.optimizers.Optimizer,
    ) -> None:
        """
        Compute gradients, optionally clip, and apply them.

        Args:
            tape: Active gradient tape that recorded ``loss``.
            loss: Scalar loss tensor to differentiate.
            variables: Trainable variables to update.
            optimizer: Optimiser to apply the (clipped) gradients.
        """
        grads = tape.gradient(loss, variables)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        optimizer.apply_gradients(zip(grads, variables))

    # ------------------------------------------------------------------
    # Trainer interface
    # ------------------------------------------------------------------

    def params(self) -> dict[str, str | int | float | bool]:
        """
        Return hyperparameters logged to MLflow at the start of training.

        Combines continuous-PPO-specific hyperparameters with model metadata
        so that every run is fully reproducible from the logged params alone.

        Returns:
            dict[str, str | int | float | bool]: Flat mapping of parameter names to
                their values, ready to pass to ``mlflow.log_params()``.
        """
        return super().params() | self.model.metadata()

    def checkpoint(self) -> None:
        """
        Persist all training state to prevent data loss on failure.

        Saves the model weights under ``<checkpoint_dir>/<timestamp>/model/``
        and all serialisable hyperparameters to
        ``<checkpoint_dir>/<timestamp>/params.json``. The timestamp-based
        subdirectory ensures successive checkpoints do not overwrite each other.
        """
        checkpoint_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self.checkpoint_dir) / checkpoint_id

        model_path = path / "model"
        model_path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(model_path))

        with open(path / "params.json", "w") as f:
            json.dump(self.params(), f, indent=2)

        logger.info(f"Checkpoint {checkpoint_id} saved to {self.checkpoint_dir}")

    def recovery(self, checkpoint_id: str) -> None:
        """
        Restore training state from a checkpoint.

        Loads model weights from ``<checkpoint_dir>/<checkpoint_id>/model`` and
        restores declared class attributes from
        ``<checkpoint_dir>/<checkpoint_id>/params.json``. Only keys present in
        the class-level annotations across the full MRO are restored; any extra
        keys in the JSON (e.g. model metadata) are ignored.

        Args:
            checkpoint_id: Identifier of the checkpoint subdirectory (timestamp
                string) produced by :meth:`checkpoint`.
        """
        allowed = {
            key
            for cls in type(self).__mro__
            for key in getattr(cls, "__annotations__", {})
        }

        path = Path(self.checkpoint_dir) / checkpoint_id
        self.model.load(str(path / "model"))

        with open(path / "params.json", "r") as f:
            params = json.load(f)
            for key, value in params.items():
                if key in allowed:
                    setattr(self, key, value)

        self.actor_optimizer.learning_rate.assign(self.actor_lr)
        self.critic_optimizer.learning_rate.assign(self.critic_lr)

        logger.info(f"Recovered training state from {path}")

    def policy(self, observations: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Sample continuous actions for a batch of observations.

        Actions are sampled from the squashed Gaussian and returned as a
        float32 array of shape ``(batch_size, action_dim)`` with values in
        ``(-1, 1)``.

        Args:
            observations: Batch of observations, shape ``(batch_size, obs_dim)``.

        Returns:
            NDArray[np.float32]: Sampled actions, shape ``(batch_size, action_dim)``.
        """
        obs_t = tf.constant(observations, dtype=tf.float32)
        actions, _ = self._sample_action(obs_t)
        return actions.numpy().astype(np.float32)

    def update(
        self,
        transitions: list[TransitionSchema],
    ) -> dict[str, float]:
        """
        Perform a full PPO update cycle on a batch of on-policy transitions.

        Extracts arrays from the transition objects, computes TD(0) advantages
        once (fixed), recovers pre-tanh raw actions via ``atanh``, computes old
        log-probabilities under the current policy (fixed), then runs
        ``ppo_epochs`` passes of mini-batch gradient descent using:

        - **Actor**: clipped surrogate objective with optional entropy bonus.
        - **Critic**: MSE loss on TD(0) value targets.

        For terminal transitions (``next_step is None``), the current
        observation is used as the next observation and the non-terminal
        mask zeroes out the bootstrap term.

        Args:
            transitions: List of
                :class:`~corl.schemas.tracker.Transition` objects
                collected since the last update.

        Returns:
            dict with keys ``"actor_loss"``, ``"critic_loss"``,
            ``"entropy"``, ``"mean_advantage"``, ``"mean_value"``,
            ``"mean_reward"``, ``"clip_fraction"``, ``"approx_kl"``.
            Values are averaged over all mini-batch steps.
        """
        observations = np.array(
            [t.current_step.observation for t in transitions], dtype=np.float32
        )
        actions = np.array(
            [t.current_step.action for t in transitions], dtype=np.float32
        )
        rewards = np.array(
            [t.current_step.reward for t in transitions], dtype=np.float32
        )
        next_observations = np.array(
            [
                t.next_step.observation if t.next_step else t.current_step.observation
                for t in transitions
            ],
            dtype=np.float32,
        )
        terminals = np.array([t.current_step.done for t in transitions], dtype=np.bool_)

        obs_t = tf.constant(observations, dtype=tf.float32)
        next_obs_t = tf.constant(next_observations, dtype=tf.float32)
        non_terminal = tf.constant(~terminals, dtype=tf.float32)
        rewards_t = tf.constant(rewards, dtype=tf.float32)
        act_t = tf.constant(actions, dtype=tf.float32)

        # Recover pre-tanh raw actions; clip avoids atanh(±1) = ±inf
        raw_actions = tf.atanh(tf.clip_by_value(act_t, -1.0 + 1e-6, 1.0 - 1e-6))

        # TD(0) targets and advantages (fixed for all epochs)
        values = tf.squeeze(self.model.critic(obs_t, training=False), axis=-1)
        next_values = tf.squeeze(self.model.critic(next_obs_t, training=False), axis=-1)
        td_targets = rewards_t + self.gamma * next_values * non_terminal
        advantages = (td_targets - values).numpy()
        td_targets_np = td_targets.numpy()

        # Old log-probabilities under the current policy (fixed for all epochs)
        old_output = self.model.actor(obs_t, training=False)
        old_mean, old_log_std = tf.split(old_output, 2, axis=-1)
        old_mean = tf.clip_by_value(old_mean, -4.0, 4.0)
        old_log_std = tf.clip_by_value(old_log_std, LOG_STD_MIN, LOG_STD_MAX)
        old_log_probs = self._gaussian_log_prob(
            raw_actions, old_mean, old_log_std
        ).numpy()

        raw_actions_np = raw_actions.numpy()
        n = len(transitions)

        accum = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
            "approx_kl": 0.0,
        }
        num_steps = 0

        for _ in range(self.ppo_epochs):
            # Normalise advantages per epoch
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages) + 1e-8
            norm_advantages = (advantages - adv_mean) / adv_std

            indices = np.random.permutation(n)

            for start in range(0, n, self.mini_batch_size):
                mb_idx = indices[start : start + self.mini_batch_size]

                mb_obs = tf.constant(observations[mb_idx])
                mb_raw = tf.constant(raw_actions_np[mb_idx], dtype=tf.float32)
                mb_old_log_probs = tf.constant(old_log_probs[mb_idx], dtype=tf.float32)
                mb_advantages = tf.constant(norm_advantages[mb_idx], dtype=tf.float32)
                mb_td_targets = tf.constant(td_targets_np[mb_idx], dtype=tf.float32)

                # Actor: clipped surrogate + entropy bonus
                with tf.GradientTape() as actor_tape:
                    new_output = self.model.actor(mb_obs, training=True)
                    new_mean, new_log_std = tf.split(new_output, 2, axis=-1)
                    new_mean = tf.clip_by_value(new_mean, -4.0, 4.0)
                    new_log_std = tf.clip_by_value(
                        new_log_std, LOG_STD_MIN, LOG_STD_MAX
                    )
                    new_log_probs = self._gaussian_log_prob(
                        mb_raw, new_mean, new_log_std
                    )

                    log_ratio = tf.clip_by_value(
                        new_log_probs - mb_old_log_probs, -10.0, 10.0
                    )
                    ratio = tf.exp(log_ratio)
                    clipped_ratio = tf.clip_by_value(
                        ratio,
                        1.0 - self.clip_range,
                        1.0 + self.clip_range,
                    )
                    policy_loss = -tf.reduce_mean(
                        tf.minimum(ratio * mb_advantages, clipped_ratio * mb_advantages)
                    )

                    # Differential entropy of N(0, σ²): 0.5*(1 + log(2πe*σ²))
                    entropy = tf.reduce_mean(
                        tf.reduce_sum(
                            0.5 * (1.0 + tf.math.log(2.0 * np.pi) + 2.0 * new_log_std),
                            axis=-1,
                        )
                    )
                    actor_loss = policy_loss - self.entropy_coeff * entropy

                self._apply_gradients(
                    actor_tape,
                    actor_loss,
                    self.model.actor.trainable_variables,
                    self.actor_optimizer,
                )

                # Critic: MSE on value targets
                with tf.GradientTape() as critic_tape:
                    pred_values = tf.squeeze(
                        self.model.critic(mb_obs, training=True), axis=-1
                    )
                    critic_loss = self.value_loss_coeff * tf.reduce_mean(
                        tf.square(mb_td_targets - pred_values)
                    )

                self._apply_gradients(
                    critic_tape,
                    critic_loss,
                    self.model.critic.trainable_variables,
                    self.critic_optimizer,
                )

                clip_frac = tf.reduce_mean(
                    tf.cast(tf.abs(ratio - 1.0) > self.clip_range, tf.float32)
                )
                approx_kl = tf.reduce_mean((ratio - 1.0) - tf.math.log(ratio))

                accum["actor_loss"] += float(actor_loss.numpy())
                accum["critic_loss"] += float(critic_loss.numpy())
                accum["entropy"] += float(entropy.numpy())
                accum["clip_fraction"] += float(clip_frac.numpy())
                accum["approx_kl"] += float(approx_kl.numpy())
                num_steps += 1

        return {k: v / max(num_steps, 1) for k, v in accum.items()} | {
            "mean_advantage": float(np.mean(advantages)),
            "mean_value": float(tf.reduce_mean(values).numpy()),
            "mean_reward": float(np.mean(rewards)),
        }

    def run(self, max_transitions: int) -> None:
        """
        Execute the continuous PPO training loop for ``max_transitions`` transitions.

        Logs hyperparameters to MLflow on a fresh run (skipped when resuming
        from a checkpoint). Repeatedly calls
        :meth:`~corl.trainer.trainer.Trainer.training_step` to collect
        environment interactions. After every ``update_frequency``
        transitions, runs a full PPO update cycle (multiple mini-batch
        epochs) on the collected batch and discards the data.

        Metrics are logged to MLflow after each update. A checkpoint is saved
        via :meth:`checkpoint` every ``checkpoint_frequency`` transitions.

        On completion, the final model is saved and the trainer is closed.

        Args:
            max_transitions: Total number of simulation steps to run.

        Raises:
            ValueError: If ``max_transitions`` is less than ``1``.
        """
        if max_transitions < 1:
            raise ValueError(
                f"Number of transitions must be >= 1, got {max_transitions}"
            )

        if self.last_checkpoint_transition == 0:
            self._mlflow_log_train_params()
            training_step_count = 0
            last_update = 0
        else:
            training_step_count = self.last_checkpoint_transition
            last_update = self.last_checkpoint_transition

        # On-policy buffer: filled, consumed once per PPO cycle, then cleared
        buffer: list[TransitionSchema] = []

        logger.info(
            f"Starting continuous PPO training at transition {training_step_count}/{max_transitions}."
        )

        while training_step_count < max_transitions:
            steps = self.training_step()

            transitions = [
                transition
                for step_key, step_result in steps
                for transition in self.transition.make(step_key.worker_id, step_result)
            ]

            for transition in transitions:
                training_step_count += 1

                if transition.current_step.done:
                    self.episode_count += 1

                buffer.append(transition)

            if (
                training_step_count - last_update >= self.update_frequency
                and len(buffer) > 0
            ):
                update_metrics = self.update(buffer)
                buffer.clear()
                last_update = training_step_count

                mlflow.log_metrics(
                    update_metrics
                    | {
                        "episode": self.episode_count,
                        "transition_per_episode": round(
                            training_step_count / self.episode_count, 2
                        )
                        if self.episode_count > 0
                        else 0.0,
                    },
                    step=training_step_count,
                )

            if (
                training_step_count - self.last_checkpoint_transition
                >= self.checkpoint_frequency
            ):
                self.last_checkpoint_transition = training_step_count
                self.checkpoint()

            logger.debug(
                f"Processed {len(steps)} steps with {len(transitions)} "
                f"transitions. {training_step_count}/{max_transitions}."
            )

        self.model.save(self.model_dir)
        self.close()
