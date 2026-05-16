import logging

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
        model_checkpoint_frequency: int,
        gamma: float,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        update_frequency: int = 64,
    ):
        """
        Initialise the A2C trainer and create optimisers.

        Args:
            config: Application configuration (experiment paths, env vars).
            model: Actor-critic model to train.
            model_checkpoint_frequency: Transitions between checkpoint saves.
            gamma: Discount factor in ``[0, 1]``.
            entropy_coeff: Weight of the entropy bonus term. Higher values
                encourage more exploration.
            value_loss_coeff: Scaling factor for the critic MSE loss.
            actor_lr: Learning rate for the actor's Adam optimiser.
            critic_lr: Learning rate for the critic's Adam optimiser.
            update_frequency: Number of transitions to collect before each
                parameter update.
        """
        super().__init__(
            model=model,
            config=config,
            model_checkpoint_frequency=model_checkpoint_frequency,
        )
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.update_frequency = update_frequency
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.transition = TransitionMemory()

    def params(self) -> dict[str, str | int | float]:
        """
        Return hyperparameters for MLflow logging.

        Merges A2C-specific hyperparameters with the model's own metadata
        (from :meth:`~corl.model.model.Model.metadata`).

        Returns:
            dict mapping parameter name → scalar value, suitable for
            ``mlflow.log_params()``.
        """
        return {
            "gamma": self.gamma,
            "entropy_coeff": self.entropy_coeff,
            "value_loss_coeff": self.value_loss_coeff,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "update_frequency": self.update_frequency,
        } | self.model.metadata()

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

    def run(self, epochs: int) -> None:
        """
        Execute the A2C training loop for ``epochs`` transitions.

        Logs hyperparameters to MLflow, then repeatedly calls
        :meth:`~corl.trainer.trainer.Trainer.training_step` to collect
        environment interactions. After every ``update_frequency``
        transitions, performs a single A2C update on the collected batch
        and discards the data (on-policy).

        Metrics are logged to MLflow after each update. A model checkpoint
        is saved every ``model_checkpoint_frequency`` transitions.

        On completion, the final model is saved via
        :meth:`~corl.model.discrete.actor_critic.ModelActorCritic.save` and the
        trainer is closed.

        Args:
            epochs: Total number of transitions to collect.

        Raises:
            ValueError: If ``epochs`` is less than ``1``.
        """
        if epochs < 1:
            raise ValueError(f"Number of epochs must be >= 1, got {epochs}")

        mlflow.log_params(self.params())

        training_step_count = 0
        episode_count = 0
        last_update = 0
        last_checkpoint = 0

        # On-policy buffer: filled, used once per update, then cleared
        buffer: list[TransitionSchema] = []

        logger.info(f"Starting A2C training for {epochs} transitions")

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
                        "episode": episode_count,
                        "transition_per_episode": round(
                            training_step_count / episode_count, 2
                        )
                        if episode_count > 100
                        else 0.0,
                    },
                    step=training_step_count,
                )

            if training_step_count - last_checkpoint >= self.model_checkpoint_frequency:
                self.model.save_weights(self.model_dir, checkpoint=True)
                last_checkpoint = training_step_count

            logger.debug(
                f"Processed {len(steps)} steps with {len(transitions)} "
                f"transitions. {training_step_count}/{epochs}."
            )

        self.model.save(self.model_dir)
        self.close()
