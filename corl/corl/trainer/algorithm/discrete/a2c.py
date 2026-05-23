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


class TrainerA2C(Trainer):
    """
    Advantage Actor-Critic (A2C) trainer.

    Implements the synchronous A2C algorithm with separate actor and critic
    networks. The actor learns a stochastic policy via the policy gradient
    with advantage estimates, while the critic learns to predict state values
    via mean-squared-error regression. Training is driven by the Backtrain
    API via the parent :class:`~corl.trainer.trainer.Trainer` class.

    Key design decisions:

    - **TD(0) advantage**: ``A(s, a) = r + γ V(s') - V(s)`` for
      non-terminal transitions, ``A(s, a) = r - V(s)`` for terminal ones.
    - **Entropy regularisation**: an entropy bonus encourages exploration
      by penalising overly deterministic policies.
    - **On-policy**: collected transitions are used for a single update
      and then discarded (no experience replay).
    - **Synchronous workers**: data is collected from all workers in
      lockstep via the parent :class:`~corl.trainer.trainer.Trainer`.

    Attributes:
        gamma: Discount factor for future rewards.
        entropy_coeff: Weight of the entropy bonus in the actor loss.
        value_loss_coeff: Scaling factor for the critic MSE loss.
        actor_lr: Learning rate for the actor optimiser.
        critic_lr: Learning rate for the critic optimiser.
        update_frequency: Number of transitions collected between parameter
            updates.
        actor_optimizer: Adam optimiser for the actor network.
        critic_optimizer: Adam optimiser for the critic network.
        transition: Helper for assembling step results into TD transitions.
    """

    gamma: float
    entropy_coeff: float
    value_loss_coeff: float
    actor_lr: float
    critic_lr: float
    update_frequency: int
    actor_optimizer: tf.keras.optimizers.Adam
    critic_optimizer: tf.keras.optimizers.Adam
    transition: TransitionMemory

    def __init__(
        self,
        config: Config,
        model: ModelActorCritic,
        checkpoint_frequency: int,
        gamma: float,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        update_frequency: int = 64,
        checkpoint_id: str | None = None,
    ):
        """
        Initialise the A2C trainer and create optimisers.

        Args:
            config: Application configuration (experiment paths, env vars).
            model: Actor-critic model to train.
            checkpoint_frequency: Transitions between checkpoint saves.
            gamma: Discount factor in ``[0, 1]``.
            entropy_coeff: Weight of the entropy bonus term. Higher values
                encourage more exploration.
            value_loss_coeff: Scaling factor for the critic MSE loss.
            actor_lr: Learning rate for the actor's Adam optimiser.
            critic_lr: Learning rate for the critic's Adam optimiser.
            update_frequency: Number of transitions to collect before each
                parameter update.
            checkpoint_id: If provided, resume training from this checkpoint
                via :meth:`recovery`. When ``None``, a fresh training session
                is started. Defaults to ``None``.
        """

        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.update_frequency = update_frequency
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
        Return hyperparameters for MLflow logging.

        Merges A2C-specific hyperparameters with the base trainer's scalar
        attributes (from :meth:`~corl.trainer.trainer.Trainer.params`) and
        the model's own metadata (from
        :meth:`~corl.model.model.Model.metadata`).

        Returns:
            dict mapping parameter name → scalar value, suitable for
            ``mlflow.log_params()``.
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

        logger.info(f"Recovered training state from {path}")

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

    def update(
        self,
        transitions: list[TransitionSchema],
    ) -> dict[str, float]:
        """
        Perform one A2C parameter update on a batch of transitions.

        Extracts observations, actions, rewards, next-observations, and
        terminal flags from the transition objects, computes TD(0)
        advantages, then updates the actor via policy gradient (with
        entropy bonus) and the critic via MSE value regression.

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
            ``"mean_reward"``.
        """
        observations = np.array(
            [t.current_step.observation for t in transitions],
            dtype=np.float32,
        )
        actions = np.array([t.current_step.action for t in transitions], dtype=np.int32)
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
        rewards_t = tf.constant(rewards, dtype=tf.float32)
        actions_t = tf.squeeze(tf.constant(actions, dtype=tf.int32), axis=-1)
        non_terminal = tf.constant(~terminals, dtype=tf.float32)

        # Compute TD(0) targets and advantages from the critic
        values = tf.squeeze(self.model.critic(obs_t, training=False), axis=-1)
        next_values = tf.squeeze(self.model.critic(next_obs_t, training=False), axis=-1)
        td_targets = rewards_t + self.gamma * next_values * non_terminal
        advantages = td_targets - values

        # Actor update: policy gradient with entropy bonus
        with tf.GradientTape() as actor_tape:
            logits = self.model.actor(obs_t, training=True)
            log_probs = tf.nn.log_softmax(logits)
            probs = tf.nn.softmax(logits)

            batch_indices = tf.range(tf.shape(actions_t)[0])
            action_indices = tf.stack([batch_indices, actions_t], axis=1)
            action_log_probs = tf.gather_nd(log_probs, action_indices)

            entropy = -tf.reduce_sum(probs * log_probs, axis=-1)
            mean_entropy = tf.reduce_mean(entropy)

            # -E[log π(a|s) · A(s,a)] - c_ent · H(π)
            actor_loss = (
                -tf.reduce_mean(action_log_probs * tf.stop_gradient(advantages))
                - self.entropy_coeff * mean_entropy
            )

        actor_grads = actor_tape.gradient(
            actor_loss, self.model.actor.trainable_variables
        )
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.model.actor.trainable_variables)
        )

        # Critic update: MSE on value predictions
        with tf.GradientTape() as critic_tape:
            pred_values = tf.squeeze(self.model.critic(obs_t, training=True), axis=-1)
            critic_loss = self.value_loss_coeff * tf.reduce_mean(
                tf.square(tf.stop_gradient(td_targets) - pred_values)
            )

        critic_grads = critic_tape.gradient(
            critic_loss, self.model.critic.trainable_variables
        )
        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.model.critic.trainable_variables)
        )

        return {
            "actor_loss": float(actor_loss.numpy()),
            "critic_loss": float(critic_loss.numpy()),
            "entropy": float(mean_entropy.numpy()),
            "mean_advantage": float(tf.reduce_mean(advantages).numpy()),
            "mean_value": float(tf.reduce_mean(values).numpy()),
            "mean_reward": float(tf.reduce_mean(rewards_t).numpy()),
        }

    def run(self, max_transitions: int) -> None:
        """
        Execute the A2C training loop for ``max_transitions`` transitions.

        Logs hyperparameters to MLflow on the first (or resumed) run, then
        repeatedly calls :meth:`~corl.trainer.trainer.Trainer.training_step`
        to collect environment interactions. After every ``update_frequency``
        transitions, performs a single A2C update on the collected batch and
        discards the data (on-policy).

        Metrics are logged to MLflow after each update. A model checkpoint
        is saved every ``checkpoint_frequency`` transitions via
        :meth:`checkpoint`.

        On completion, the final model is saved via
        :meth:`~corl.model.discrete.actor_critic.ModelActorCritic.save` and the
        trainer is closed.

        Args:
            max_transitions: Total number of transitions to collect.

        Raises:
            ValueError: If ``epochs`` is less than ``1``.
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

        buffer: list[TransitionSchema] = []

        logger.info(
            f"Starting A2C training at transition {training_step_count}/{max_transitions}."
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
