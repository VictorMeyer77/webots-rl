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


class TrainerPPO(Trainer):
    """
    Proximal Policy Optimisation (PPO) trainer with clipped surrogate objective.

    Implements the PPO-Clip algorithm with separate actor and critic
    networks. Collected on-policy data is reused for multiple mini-batch
    gradient steps, constrained by a clipped probability-ratio objective
    that prevents destructively large policy updates. Training is driven
    by the Backtrain API via the parent
    :class:`~corl.trainer.trainer.Trainer` class.

    Key design decisions:

    - **Clipped surrogate objective**: the policy ratio
      ``r(θ) = π_new(a|s) / π_old(a|s)`` is clipped to
      ``[1 - clip_range, 1 + clip_range]`` so the effective update is
      bounded regardless of the advantage sign.
    - **Multiple mini-batch epochs**: the same rollout buffer is split
      into random mini-batches and iterated over ``ppo_epochs`` times,
      amortising the cost of environment interaction.
    - **Advantage normalisation**: advantages are standardised to zero
      mean and unit variance before each epoch, stabilising training.
    - **Entropy regularisation**: an entropy bonus discourages premature
      policy collapse.
    - **On-policy**: after all mini-batch epochs the buffer is discarded.
    - **Synchronous workers**: data is collected from all workers in
      lockstep via the parent :class:`~corl.trainer.trainer.Trainer`.

    Attributes:
        gamma: Discount factor for future rewards.
        clip_range: Clipping parameter ε for the surrogate objective.
        ppo_epochs: Number of optimisation passes over each collected
            rollout.
        mini_batch_size: Number of transitions per mini-batch. The rollout
            is randomly shuffled and split into chunks of this size.
        entropy_coeff: Weight of the entropy bonus in the actor loss.
        value_loss_coeff: Scaling factor for the critic MSE loss.
        actor_lr: Learning rate for the actor optimiser.
        critic_lr: Learning rate for the critic optimiser.
        update_frequency: Number of transitions collected between parameter
            updates.
        max_grad_norm: Maximum L2 norm for gradient clipping. ``None``
            disables clipping.
        actor_optimizer: Adam optimiser for the actor network.
        critic_optimizer: Adam optimiser for the critic network.
        transition: Helper for assembling step results into TD transitions.
    """

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
        model: ModelActorCritic,
        checkpoint_frequency: int,
        gamma: float,
        clip_range: float = 0.2,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        update_frequency: int = 256,
        max_grad_norm: float | None = 0.5,
        checkpoint_id: str | None = None,
    ):
        """
        Initialise the PPO trainer and create optimisers.

        Args:
            config: Application configuration (experiment paths, env vars).
            model: Actor-critic model to train.
            checkpoint_frequency: Transitions between checkpoint saves.
            gamma: Discount factor in ``[0, 1]``.
            clip_range: Clipping parameter ε for the surrogate objective.
                Typical values are ``0.1``–``0.3``.
            ppo_epochs: Number of passes over the rollout buffer per
                update cycle.
            mini_batch_size: Number of transitions per mini-batch within
                each PPO epoch.
            entropy_coeff: Weight of the entropy bonus term. Higher values
                encourage more exploration.
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

        super().__init__(
            model=model,
            config=config,
            checkpoint_frequency=checkpoint_frequency,
            checkpoint_id=checkpoint_id,
        )

    def params(self) -> dict[str, str | int | float | bool]:
        """
        Return hyperparameters logged to MLflow at the start of training.

        Combines PPO-specific hyperparameters with model metadata so that every
        run is fully reproducible from the logged params alone.

        Returns:
            dict[str, str | int | float | bool]: Flat mapping of parameter names to
                their values, ready to pass to ``mlflow.log_params()``.
        """
        return super().params() | self.model.metadata()

    def policy(self, observations: NDArray[np.float32]) -> NDArray[np.int32]:
        """
        Sample actions from the actor's stochastic policy.

        Delegates to
        :meth:`~corl.model.discrete.actor_critic.ModelActorCritic.predict` which
        samples from the categorical distribution defined by the actor's
        softmax output.

        Args:
            observations: Batch of observations, shape ``(batch_size, obs_dim)``.

        Returns:
            NDArray[np.int32]: Sampled action indices, shape ``(batch_size,)``.
        """
        return self.model.predict(observations)

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

        logger.info(f"Recovered training state from {path}")

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

    def update(
        self,
        transitions: list[TransitionSchema],
    ) -> dict[str, float]:
        """
        Perform a full PPO update cycle on a batch of transitions.

        Extracts arrays from the transition objects, computes TD(0)
        advantages and old log-probabilities once, then runs
        ``ppo_epochs`` passes of mini-batch gradient descent using the
        clipped surrogate objective for the actor and MSE for the critic.

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
            [t.current_step.observation for t in transitions],
            dtype=np.float32,
        )
        actions = np.array(
            [t.current_step.action for t in transitions], dtype=np.int32
        ).squeeze(-1)
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

        n = len(transitions)
        obs_t = tf.constant(observations, dtype=tf.float32)
        next_obs_t = tf.constant(next_observations, dtype=tf.float32)
        non_terminal = tf.constant(~terminals, dtype=tf.float32)

        # Compute TD(0) targets and advantages (fixed for all epochs)
        rewards_t = tf.constant(rewards, dtype=tf.float32)
        values = tf.squeeze(self.model.critic(obs_t, training=False), axis=-1)
        next_values = tf.squeeze(self.model.critic(next_obs_t, training=False), axis=-1)
        td_targets = rewards_t + self.gamma * next_values * non_terminal
        advantages = (td_targets - values).numpy()

        # Old log-probabilities (fixed for all epochs)
        old_logits = self.model.actor(obs_t, training=False)
        old_log_probs_all = tf.nn.log_softmax(old_logits).numpy()
        old_action_log_probs = old_log_probs_all[np.arange(n), actions]

        td_targets_np = td_targets.numpy()

        # Mini-batch PPO epochs
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
                mb_actions = tf.constant(actions[mb_idx], dtype=tf.int32)
                mb_old_log_probs = tf.constant(
                    old_action_log_probs[mb_idx], dtype=tf.float32
                )
                mb_advantages = tf.constant(norm_advantages[mb_idx], dtype=tf.float32)
                mb_td_targets = tf.constant(td_targets_np[mb_idx], dtype=tf.float32)

                # Actor: clipped surrogate objective
                with tf.GradientTape() as actor_tape:
                    logits = self.model.actor(mb_obs, training=True)
                    log_probs = tf.nn.log_softmax(logits)
                    probs = tf.nn.softmax(logits)

                    mb_batch_idx = tf.range(tf.shape(mb_actions)[0])
                    action_idx = tf.stack([mb_batch_idx, mb_actions], axis=1)
                    new_log_probs = tf.gather_nd(log_probs, action_idx)

                    ratio = tf.exp(new_log_probs - mb_old_log_probs)
                    clipped_ratio = tf.clip_by_value(
                        ratio,
                        1.0 - self.clip_range,
                        1.0 + self.clip_range,
                    )

                    surr1 = ratio * mb_advantages
                    surr2 = clipped_ratio * mb_advantages
                    policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

                    entropy = -tf.reduce_sum(probs * log_probs, axis=-1)
                    mean_entropy = tf.reduce_mean(entropy)

                    actor_loss = policy_loss - self.entropy_coeff * mean_entropy

                self._apply_gradients(
                    actor_tape,
                    actor_loss,
                    self.model.actor.trainable_variables,
                    self.actor_optimizer,
                )

                # Critic: MSE on value predictions
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

                # Diagnostics
                clip_frac = tf.reduce_mean(
                    tf.cast(tf.abs(ratio - 1.0) > self.clip_range, tf.float32)
                )
                approx_kl = tf.reduce_mean((ratio - 1.0) - tf.math.log(ratio))

                accum["actor_loss"] += float(actor_loss.numpy())
                accum["critic_loss"] += float(critic_loss.numpy())
                accum["entropy"] += float(mean_entropy.numpy())
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
        Execute the PPO training loop for ``max_transitions`` transitions.

        Logs hyperparameters to MLflow, then repeatedly calls
        :meth:`~corl.trainer.trainer.Trainer.training_step` to collect
        environment interactions. After every ``update_frequency``
        transitions, runs a full PPO update cycle (multiple mini-batch
        epochs) on the collected batch and discards the data.

        Metrics are logged to MLflow after each update. A model checkpoint
        is saved every ``checkpoint_frequency`` transitions.

        On completion, the final model is saved via
        :meth:`~corl.model.discrete.actor_critic.ModelActorCritic.save` and the
        trainer is closed.

        Args:
            max_transitions: Total number of transitions to collect.

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

        # On-policy buffer: filled, used for one PPO cycle, then cleared
        buffer: list[TransitionSchema] = []

        logger.info(
            f"Starting training at transition {training_step_count}/{max_transitions}."
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

            # Update when enough transitions have been collected
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
