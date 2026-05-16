import logging
from abc import abstractmethod

import mlflow
import numpy as np
from numpy.typing import NDArray

from corl.memory.transition import Transition as TransitionMemory
from corl.model.discrete.value_table import ModelValueTable
from corl.schemas.tracker import Transition as TransitionSchema
from corl.trainer.trainer import Trainer
from corl.utils.config import Config

logger = logging.getLogger(__name__)


class TrainerTDTabular(Trainer):
    """
    Abstract base class for tabular temporal-difference (TD) trainers.

    Implements the shared training loop, ε-greedy policy, and metric logging
    for TD algorithms that operate on a discrete value table. Concrete
    subclasses — :class:`~corl.trainer.algorithm.discrete.q_learning.TrainerQLearning`
    (off-policy) and :class:`~corl.trainer.algorithm.discrete.sarsa.TrainerSarsa`
    (on-policy) — only need to implement :meth:`update_value_table` with their
    specific TD target formula.

    Attributes:
        alpha (float): Learning rate — controls how much each TD update shifts
            the current Q-value towards the TD target.
        gamma (float): Discount factor — weights the importance of future
            rewards relative to immediate rewards (0 = myopic, 1 = far-sighted).
        epsilon (float): Current exploration probability for the ε-greedy
            policy. Decayed once per transition.
        epsilon_min (float): Lower bound for epsilon; decay stops here.
        epsilon_decay (float): Multiplicative factor applied to epsilon once
            per transition (e.g. ``0.9999``).
        transition (TransitionMemory): Stateful buffer that pairs consecutive
            steps into :class:`~corl.schemas.tracker.Transition` objects
            suitable for TD updates.
    """

    alpha: float
    gamma: float
    epsilon: float
    epsilon_decay: float
    epsilon_min: float
    transition: TransitionMemory

    def __init__(
        self,
        config: Config,
        model: ModelValueTable,
        model_checkpoint_frequency: int,
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
    ):
        """
        Initialise the TD tabular trainer.

        Args:
            config (Config): Application configuration used by the parent
                :class:`~corl.trainer.trainer.Trainer`.
            model (ModelValueTable): Discrete value-table model that stores and
                updates Q-values.
            model_checkpoint_frequency (int): Minimum number of transitions
                between automatic model checkpoints.
            alpha (float): Learning rate for TD updates.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Initial exploration probability for ε-greedy
                action selection.
            epsilon_min (float): Minimum value epsilon can decay to.
            epsilon_decay (float): Multiplicative factor applied to epsilon
                once per transition.
        """
        super().__init__(
            model=model,
            config=config,
            model_checkpoint_frequency=model_checkpoint_frequency,
        )
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.transition = TransitionMemory()

    def params(self) -> dict[str, str | int | float]:
        """
        Return hyperparameters logged to MLflow at the start of training.

        Combines TD-specific hyperparameters with model metadata so that every
        run is fully reproducible from the logged params alone.

        Returns:
            dict[str, str | int | float]: Flat mapping of parameter names to
                their values, ready to pass to ``mlflow.log_params()``.
        """
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_initial": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        } | self.model.metadata()

    def policy(self, observations: NDArray[np.float32]) -> NDArray[np.int32]:
        """
        Select actions for a batch of observations using the ε-greedy policy.

        Args:
            observations: Array of shape ``(N, obs_dim)`` containing the
                current observations for ``N`` workers.

        Returns:
            Integer action array of shape ``(N,)``, one action per observation.
        """
        return np.array(
            [
                self.model.epsilon_greedy_policy(observation, self.epsilon)
                for observation in observations
            ]
        )

    @abstractmethod
    def update_value_table(self, transition: TransitionSchema) -> dict[str, float]:
        """
        Apply a single TD update to the value table and return the TD error.

        Subclasses implement the algorithm-specific TD target:

        - **Q-learning** (off-policy): ``r + γ · max_a Q(s', a)``
        - **SARSA** (on-policy): ``r + γ · Q(s', a')``

        Args:
            transition (TransitionSchema): A consecutive pair of steps
                ``(current_step, next_step)`` produced by
                :attr:`transition`.

        Returns:
            dict[str, float]: Metrics from the update, typically including
                ``td_error`` and ``reward``, merged with shared metrics
                (``episode``, ``transition_per_episode``,
                ``value_table_nonzero``, ``epsilon``) before being logged
                to MLflow.
        """

    def run(self, epochs: int) -> None:
        """
        Run the full TD training loop.

        Iterates until ``epochs`` total transitions have been processed.
        Each outer iteration calls :meth:`~corl.trainer.trainer.Trainer.training_step`
        to collect a batch of steps, then for every transition in the batch:

        1. Calls :meth:`update_value_table` to apply the TD update.
        2. Logs metrics to MLflow every ``log_metric_frequency`` transitions
           (only when ``update_value_table`` returns a non-empty dict):
           subclass metrics merged with ``episode``,
           ``transition_per_episode``, ``value_table_nonzero``, ``epsilon``.
        3. Saves a weight checkpoint every ``model_checkpoint_frequency``
           transitions.
        4. Decays ``epsilon`` by ``epsilon_decay`` once per transition
           (floored at ``epsilon_min``).

        After all transitions, saves the final model and closes the trainer.

        Args:
            epochs: Total number of transitions to process before stopping.
                Must be >= 1.

        Raises:
            ValueError: If ``epochs < 1``.
        """
        if epochs < 1:
            raise ValueError(f"Number of transitions must be >= 1, got {epochs}")

        mlflow.log_params(self.params())

        training_step_count = 0
        episode_count = 0
        last_checkpoint = 0
        last_metric_log = 0

        logger.info(f"Starting training for {epochs} transitions")

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

                fit_metrics = self.update_value_table(transition)

                if (
                    fit_metrics
                    and training_step_count - last_metric_log
                    >= self.log_metric_frequency
                ):
                    mlflow.log_metrics(
                        fit_metrics
                        | {
                            "episode": episode_count,
                            "transition_per_episode": round(
                                training_step_count / episode_count, 2
                            )
                            if episode_count > 100
                            else 0.0,
                            "value_table_nonzero": int(
                                np.count_nonzero(self.model.value_table)
                            ),
                            "epsilon": self.epsilon,
                        },
                        step=training_step_count,
                    )
                    last_metric_log = training_step_count

            if training_step_count - last_checkpoint >= self.model_checkpoint_frequency:
                self.model.save_weights(self.model_dir, checkpoint=True)
                last_checkpoint = training_step_count

            for _ in transitions:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            logger.debug(
                f"Processed {len(steps)} steps with {len(transitions)}. {training_step_count}/{epochs}."
            )

        self.model.save(self.model_dir)
        self.close()
