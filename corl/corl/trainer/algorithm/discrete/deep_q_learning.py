import json
import logging
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from corl.memory.prioritized_experience_replay import PrioritizedExperienceReplayBuffer
from corl.memory.transition import Transition as TransitionMemory
from corl.model.discrete.deep_value_table import ModelDeepValueTable
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
      annealed from ``per_beta_start`` to ``1.0`` over ``max_transitions`` steps.

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
        checkpoint_frequency: int,
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
        checkpoint_id: str | None = None,
    ):
        """
        Initialise the DDQN trainer and create the target network.

        Args:
            config: Application configuration (experiment paths, env vars).
            model: Online Q-network to train.
            checkpoint_frequency: Steps between checkpoint saves.
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
            checkpoint_id: If provided, resume training from this checkpoint
                via :meth:`recovery`. When ``None``, a fresh training
                session is started. Defaults to ``None``.
        """
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

        super().__init__(
            model=model,
            config=config,
            checkpoint_frequency=checkpoint_frequency,
            checkpoint_id=checkpoint_id,
        )

    def params(self) -> dict[str, str | int | float | bool]:
        """
        Return hyperparameters logged to MLflow at the start of training.

        Combines DDQN-specific hyperparameters with model metadata so that every
        run is fully reproducible from the logged params alone.

        Returns:
            dict[str, str | int | float | bool]: Flat mapping of parameter names to
                their values, ready to pass to ``mlflow.log_params()``.
        """
        return super().params() | self.model.metadata()

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

    def checkpoint(self) -> None:
        """
        Persist all training state to prevent data loss on failure.

        Saves the model weights under ``<checkpoint_dir>/<timestamp>/model/``,
        the target network to ``<checkpoint_dir>/<timestamp>/target_weights.keras``,
        and all serialisable hyperparameters to ``<checkpoint_dir>/<timestamp>/params.json``.
        The timestamp-based subdirectory ensures successive checkpoints do not
        overwrite each other.
        """
        checkpoint_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self.checkpoint_dir) / checkpoint_id

        model_path = path / "model"
        model_path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(model_path))

        target_path = path / "target_weights.keras"
        self.target_weights.save(target_path)

        with open(path / "params.json", "w") as f:
            json.dump(self.params(), f, indent=2)

        logger.info(f"Checkpoint {checkpoint_id} saved to {self.checkpoint_dir}")

    def recovery(self, checkpoint_id: str) -> None:
        """
        Restore training state from a checkpoint.

        Loads model weights from ``<checkpoint_dir>/<checkpoint_id>/model``,
        the target network from ``<checkpoint_dir>/<checkpoint_id>/target_weights.keras``,
        and restores declared class attributes from ``<checkpoint_dir>/<checkpoint_id>/params.json``.
        Only keys present in the class-level annotations across the full MRO are
        restored; any extra keys in the JSON (e.g. model metadata) are ignored.

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

        target_path = path / "target_weights.keras"
        self.target_weights = tf.keras.models.load_model(target_path)

        with open(path / "params.json", "r") as f:
            params = json.load(f)
            for key, value in params.items():
                if key in allowed:
                    setattr(self, key, value)

        logger.info(f"Recovered training state from {path}")

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
        actions = actions.squeeze(
            -1
        )  # (batch_size, 1) -> (batch_size,) for discrete indexing

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

    def run(self, max_transitions: int) -> None:
        """
        Execute the full DDQN training loop for ``max_transitions`` steps.

        Logs hyperparameters to MLflow on a fresh run (skipped when resuming
        from a checkpoint). Repeatedly calls
        :meth:`~corl.trainer.trainer.Trainer.training_step` to collect
        environment interactions. After each step batch:

        - Transitions are pushed into the PER buffer.
        - :meth:`fit_model` is called every ``fit_frequency`` steps;
          metrics are logged to MLflow when a fit occurs.
        - :meth:`update_target_weights` is called every
          ``update_target_weights_frequency`` steps.
        - A checkpoint is saved via :meth:`checkpoint` every
          ``checkpoint_frequency`` steps.
        - ``epsilon`` and ``per_beta`` are updated once per transition.

        On completion, the final model is saved via
        :meth:`~corl.model.deep_value_table.ModelDeepValueTable.save`
        and the trainer is closed.

        Args:
            max_transitions: Total number of simulation steps to run.

        Raises:
            ValueError: If ``max_transitions`` is less than ``1``.
        """
        if max_transitions < 1:
            raise ValueError(f"max_transitions must be >= 1, got {max_transitions}")

        if self.last_checkpoint_transition == 0:
            self._mlflow_log_train_params()
            training_step_count = 0
            last_fit = 0
            last_target_update = 0
        else:
            training_step_count = self.last_checkpoint_transition
            last_fit = self.last_checkpoint_transition
            last_target_update = self.last_checkpoint_transition

        self.per_beta_increment = (1.0 - self.per_beta) / (
            max_transitions - training_step_count
        )

        logger.info(
            f"Starting training at transition {training_step_count}/{max_transitions}, beta increment: {self.per_beta_increment}"
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
                            "episode": self.episode_count,
                            "transition_per_episode": round(
                                training_step_count / self.episode_count, 2
                            )
                            if self.episode_count > 0
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

            if (
                training_step_count - self.last_checkpoint_transition
                >= self.checkpoint_frequency
            ):
                self.last_checkpoint_transition = training_step_count
                self.checkpoint()

            for _ in transitions:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

            logger.debug(
                f"Processed {len(steps)} steps with {len(transitions)}. {training_step_count}/{max_transitions}."
            )

        self.model.save(self.model_dir)
        self.close()
