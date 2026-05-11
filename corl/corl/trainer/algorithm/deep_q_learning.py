import logging

import mlflow
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from corl.memory.prioritized_experience_replay import PrioritizedExperienceReplayBuffer
from corl.memory.transition import Transition as TransitionMemory
from corl.model.deep_value_table import ModelDeepValueTable
from corl.trainer.trainer import Trainer
from corl.utils.config import Config

logger = logging.getLogger(__name__)


class TrainerDeepQLearning(Trainer):
    """
    Double Deep Q-Network (DDQN) trainer with Prioritized Experience Replay.

    Implements the Double DQN training loop where the **online network**
    selects the greedy next action and the **target network** evaluates it,
    reducing the maximisation bias present in standard DQN. Uses a
    :class:`~corl.memory.prioritized_experience_replay.PrioritizedExperienceReplayBuffer`
    for efficient experience sampling. Training is driven by the Backtrain
    API via the parent :class:`~corl.trainer.trainer.Trainer` class.

    Key design decisions:

    - **Target network**: a frozen copy of the online network updated
      periodically (every ``update_target_weights_frequency`` steps) to
      reduce the moving-target problem.
    - **PER**: experiences are sampled proportionally to their TD-error;
      importance-sampling weights correct the resulting bias.
    - **ε-greedy exploration**: epsilon decays multiplicatively per
      transition down to ``epsilon_min``.
    - **β annealing**: the PER importance-sampling exponent is linearly
      annealed from ``per_beta_start`` to ``1.0`` over ``epochs`` steps.

    Attributes:
        gamma: Discount factor for future rewards.
        epsilon: Current exploration rate (decays during training).
        epsilon_min: Lower bound for ``epsilon``.
        epsilon_decay: Multiplicative decay applied to ``epsilon`` after
            each transition.
        batch_size: Number of experiences sampled per training step.
        fit_frequency: Minimum number of steps between model fitting calls.
        update_target_weights_frequency: Steps between target network syncs.
        target_weights: Frozen copy of the online Keras model used for
            Bellman target computation.
        per_beta: Current importance-sampling exponent (annealed toward
            ``1.0``).
        per_beta_increment: Per-transition increment added to ``per_beta``,
            set at the start of :meth:`run`.
        transition: Helper for assembling raw step results into
            :class:`~corl.memory.transition.Transition` objects.
        experience_replay: PER buffer storing and sampling transitions.
    """

    gamma: float
    epsilon: float
    epsilon_min: float
    epsilon_decay: float
    batch_size: int
    fit_frequency: int
    update_target_weights_frequency: int
    target_weights: tf.keras.Model
    per_beta: float
    per_beta_increment: float
    transition: TransitionMemory
    experience_replay: PrioritizedExperienceReplayBuffer

    def __init__(
        self,
        config: Config,
        model: ModelDeepValueTable,
        model_checkpoint_frequency: int,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        batch_size: int,
        fit_frequency: int,
        update_target_weights_frequency: int,
        per_size: int,
        per_alpha: float,
        per_beta_start: float,
    ):
        """
        Initialise the DDQN trainer and create the target network.

        Args:
            config: Application configuration (experiment paths, env vars).
            model: Online Q-network to train.
            model_checkpoint_frequency: Steps between checkpoint saves.
            gamma: Discount factor in ``[0, 1]``.
            epsilon: Initial exploration rate.
            epsilon_min: Minimum value ``epsilon`` can decay to.
            epsilon_decay: Multiplicative factor applied to ``epsilon``
                after each transition.
            batch_size: Number of transitions to sample per fit call.
            fit_frequency: Steps between calls to :meth:`fit_model`.
            update_target_weights_frequency: Steps between calls to
                :meth:`update_target_weights`.
            per_size: Capacity of the PER replay buffer.
            per_alpha: PER priority exponent. ``0`` = uniform, ``1`` =
                full prioritisation.
            per_beta_start: Initial IS correction exponent, annealed to
                ``1.0`` over the course of training.
        """
        super().__init__(
            model=model,
            config=config,
            model_checkpoint_frequency=model_checkpoint_frequency,
        )
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.fit_frequency = fit_frequency
        self.update_target_weights_frequency = update_target_weights_frequency
        self.per_beta = per_beta_start
        self.transition = TransitionMemory()
        self.experience_replay = PrioritizedExperienceReplayBuffer(
            capacity=per_size, alpha=per_alpha
        )
        self.target_weights = tf.keras.models.clone_model(model.weights)
        self.target_weights.set_weights(model.weights.get_weights())

    def params(self) -> dict[str, str | int | float]:
        """
        Return hyperparameters for MLflow logging.

        Merges DDQN-specific hyperparameters with the model's own metadata
        (from :meth:`~corl.model.model.Model.metadata`).

        Returns:
            dict mapping parameter name → scalar value, suitable for
            ``mlflow.log_params()``.
        """
        return {
            "gamma": self.gamma,
            "epsilon_initial": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "fit_frequency": self.fit_frequency,
            "update_target_weights_frequency": self.update_target_weights_frequency,
            "per_alpha": self.experience_replay.alpha,
            "per_beta": self.per_beta,
        } | self.model.metadata()

    def policy(self, observations: NDArray[np.float32]) -> NDArray[np.int32]:
        """
        Select actions for a batch of observations using the current ε-greedy policy.

        Delegates to
        :meth:`~corl.model.deep_value_table.ModelDeepValueTable.epsilon_greedy_policy`
        with the current ``epsilon``.

        Args:
            observations: Batch of observations, shape ``(batch_size, obs_dim)``.

        Returns:
            NDArray[np.int32]: Selected action indices, shape ``(batch_size,)``.
        """
        return self.model.epsilon_greedy_policy(observations, self.epsilon)

    def update_target_weights(self) -> None:
        """
        Copy the online network's weights into the target network.

        Called periodically during training (every
        ``update_target_weights_frequency`` steps) to refresh the frozen
        Bellman bootstrap target.
        """
        self.target_weights.set_weights(self.model.weights.get_weights())
        logger.debug("Target model weights updated from training model")

    def fit_model(self) -> dict[str, float] | None:
        """
        Sample a batch from the PER buffer and perform one gradient update.

        Returns ``None`` immediately if the buffer holds fewer experiences
        than ``batch_size``. Otherwise:

        1. Samples a stratified batch from
           :attr:`experience_replay` with the current ``per_beta``.
        2. Computes Double DQN Bellman targets: the **online network**
           selects the best next action
           ``a* = argmax Q_online(s', ·)`` and the **target network**
           evaluates it:
           ``y = r + γ · Q_target(s', a*) · (1 - done)``.
        3. Fits the online network for one epoch with PER importance-
           sampling weights as ``sample_weight``.
        4. Updates PER priorities using the pre-fit TD-errors
           (no extra forward pass required).

        Returns:
            dict with keys ``"loss"``, ``"mean_td_error"``,
            ``"mean_reward"``, ``"mean_max_next_q"`` on success, or
            ``None`` if the buffer is not yet full enough to sample.
        """
        if len(self.experience_replay) < self.batch_size:
            return None

        observations, actions, rewards, next_observations, terminals, idxs, weights = (
            self.experience_replay.sample(self.batch_size, self.per_beta)
        )

        logger.debug(
            f"observations shape: {observations.shape} next_observations shape: {next_observations.shape} "
            f"actions shape: {actions.shape} rewards shape: {rewards.shape} terminals shape: {terminals.shape}"
        )

        # Double DQN: online network selects, target network evaluates
        online_next_q = self.model.weights(next_observations, training=False).numpy()
        best_actions = np.argmax(online_next_q, axis=1)
        target_next_q = self.target_weights(next_observations, training=False).numpy()
        max_next_q = target_next_q[np.arange(self.batch_size), best_actions]
        non_terminal = ~terminals

        current_q = self.model.weights(observations, training=False).numpy()
        td_targets = rewards + self.gamma * max_next_q * non_terminal

        target_q = current_q.copy()
        target_q[np.arange(self.batch_size), actions] = td_targets

        history = self.model.weights.fit(
            observations, target_q, epochs=1, verbose=0, sample_weight=weights
        )

        # TD errors use pre-fit Q values so no extra forward pass is needed
        td_errors = td_targets - current_q[np.arange(self.batch_size), actions]

        self.experience_replay.update_priorities(idxs, td_errors)

        return {
            "loss": history.history["loss"][0],
            "mean_td_error": float(np.mean(np.abs(td_errors))),
            "mean_reward": float(np.mean(rewards)),
            "mean_max_next_q": float(np.mean(max_next_q)),
        }

    def run(self, epochs: int) -> None:
        """
        Execute the full DDQN training loop for ``epochs`` steps.

        Logs hyperparameters to MLflow, then repeatedly calls
        :meth:`~corl.trainer.trainer.Trainer.training_step` to collect
        environment interactions. After each step batch:

        - Transitions are pushed into the PER buffer.
        - :meth:`fit_model` is called every ``fit_frequency`` steps;
          metrics are logged to MLflow when a fit occurs.
        - :meth:`update_target_weights` is called every
          ``update_target_weights_frequency`` steps.
        - A model checkpoint is saved every
          ``model_checkpoint_frequency`` steps.
        - ``epsilon`` and ``per_beta`` are updated once per transition.

        On completion, the final model is saved via
        :meth:`~corl.model.deep_value_table.ModelDeepValueTable.save`
        and the trainer is closed.

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
        last_target_update = 0
        last_checkpoint = 0
        self.per_beta_increment = (1.0 - self.per_beta) / epochs

        logger.info(
            f"Starting training for {epochs} transitions, beta increment: {self.per_beta_increment}"
        )

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

                self.experience_replay.add(
                    transition.current_step.observation,
                    transition.current_step.action,
                    transition.current_step.reward,
                    transition.next_step.observation
                    if transition.next_step
                    else transition.current_step.observation,
                    transition.current_step.done,
                )

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
                            if episode_count > 100
                            else 0.0,
                            "epsilon": self.epsilon,
                            "per_size": len(self.experience_replay),
                            "per_beta": self.per_beta,
                        },
                        step=training_step_count,
                    )
                last_fit = training_step_count

            if (
                training_step_count - last_target_update
                >= self.update_target_weights_frequency
            ):
                self.update_target_weights()
                last_target_update = training_step_count

            if training_step_count - last_checkpoint >= self.model_checkpoint_frequency:
                self.model.save_weights(self.model_dir, checkpoint=True)
                last_checkpoint = training_step_count

            for _ in transitions:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

            logger.debug(
                f"Processed {len(steps)} steps with {len(transitions)}. {training_step_count}/{epochs}."
            )

        self.model.save(self.model_dir)
        self.close()
