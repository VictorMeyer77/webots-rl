import collections
import logging
from abc import abstractmethod

import mlflow
import numpy as np
from numpy.typing import NDArray

from corl.memory.transition import Transition as TransitionMemory
from corl.model.value_table import ModelValueTable
from corl.schemas.learning import Observation
from corl.schemas.tracker import StepKey
from corl.schemas.tracker import Transition as TransitionSchema
from corl.trainer.trainer import Trainer
from corl.utils.config import Config

logger = logging.getLogger(__name__)

METRICS_WINDOW = 100  # Rolling window size for td_error and reward averages


class TrainerTDTabular(Trainer):
    """
    Abstract base class for tabular temporal-difference (TD) trainers.

    Implements the shared training loop, ε-greedy policy, and metric logging
    for TD algorithms that operate on a discrete value table. Concrete
    subclasses — :class:`~corl.trainer.algorithm.q_learning.TrainerQLearning`
    (off-policy) and :class:`~corl.trainer.algorithm.sarsa.TrainerSarsa`
    (on-policy) — only need to implement :meth:`update_value_table` with their
    specific TD target formula.

    Attributes:
        alpha (float): Learning rate — controls how much each TD update shifts
            the current Q-value towards the TD target.
        gamma (float): Discount factor — weights the importance of future
            rewards relative to immediate rewards (0 = myopic, 1 = far-sighted).
        epsilon (float): Current exploration probability for the ε-greedy
            policy. Decayed after every completed episode.
        epsilon_min (float): Lower bound for epsilon; decay stops here.
        epsilon_decay (float): Multiplicative factor applied to epsilon at the
            end of each episode (e.g. ``0.995``).
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
            model_checkpoint_frequency (int): Number of completed episodes
                between automatic model checkpoints.
            alpha (float): Learning rate for TD updates.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Initial exploration probability for ε-greedy
                action selection.
            epsilon_min (float): Minimum value epsilon can decay to.
            epsilon_decay (float): Multiplicative decay applied to epsilon
                after each completed episode.
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
    def parse_observations(
        self, observations: list[tuple[StepKey, Observation]]
    ) -> list[tuple[StepKey, NDArray[np.float32]]]:
        """
        Convert raw observations into numpy arrays for the model.

        Args:
            observations: List of ``(StepKey, Observation)`` pairs received
                from the environment.

        Returns:
            List of ``(StepKey, NDArray[np.float32])`` pairs containing the
            numerical representation of each observation.
        """

    @abstractmethod
    def update_value_table(self, transition: TransitionSchema) -> float:
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
            float: The TD error ``δ = td_target − Q(s, a)`` before the update
                is applied, used for logging and diagnostics.
        """

    def _log_metrics(
        self,
        epoch: int,
        accumulated_td_errors: collections.deque[float],
        accumulated_rewards: collections.deque[float],
    ) -> None:
        """
        Log rolling training metrics to MLflow.

        Computes statistics over the last :data:`METRICS_WINDOW` episodes and
        logs them as MLflow metrics at the given epoch step.

        Args:
            epoch (int): Current episode index, used as the MLflow step.
            accumulated_td_errors (deque[float]): Rolling buffer of recent TD
                errors (absolute values are averaged).
            accumulated_rewards (deque[float]): Rolling buffer of recent
                per-step rewards.
        """
        mlflow.log_metrics(
            {
                "td_error_avg": float(np.mean(np.abs(accumulated_td_errors))),
                "reward_avg": float(np.mean(accumulated_rewards)),
                "value_table_nonzero": int(np.count_nonzero(self.model.value_table)),
                "epsilon": self.epsilon,
            },
            step=epoch,
        )

    def run(self, epochs: int) -> None:
        """
        Execute the full training loop for a given number of episodes.

        Each iteration collects steps from all workers, converts them into
        transitions, applies TD updates, and logs metrics. Epsilon is decayed
        once per completed episode (``done=True`` step). A model checkpoint is
        saved every :attr:`model_checkpoint_frequency` episodes, and the final
        model is saved when all epochs are complete.

        Args:
            epochs (int): Total number of episodes to train for. Must be ≥ 1.

        Raises:
            ValueError: If ``epochs < 1``.
        """
        if epochs < 1:
            raise ValueError(f"Number of epochs must be >= 1, got {epochs}")

        mlflow.log_params(self.params())

        epoch = 0
        accumulated_td_errors: collections.deque[float] = collections.deque(
            maxlen=METRICS_WINDOW
        )
        accumulated_rewards: collections.deque[float] = collections.deque(
            maxlen=METRICS_WINDOW
        )

        logger.info(f"Starting training for {epochs} episodes")

        while epoch < epochs:
            steps = self.training_step()

            transitions = [
                transition
                for step_key, step_result in steps
                for transition in self.transition.make(step_key.worker_id, step_result)
            ]

            for transition in transitions:
                accumulated_td_errors.append(self.update_value_table(transition))

            accumulated_rewards.extend(t.current_step.reward for t in transitions)

            dones = sum(transition.current_step.done for transition in transitions)

            if dones > 0:
                self._log_metrics(
                    epoch=epoch,
                    accumulated_td_errors=accumulated_td_errors,
                    accumulated_rewards=accumulated_rewards,
                )

            for _ in range(dones):
                epoch += 1
                if epoch % self.model_checkpoint_frequency == 0:
                    self.model.save_weights(self.model_dir, checkpoint=True)
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            logger.debug(
                f"Processed {len(steps)} steps with {len(transitions)} transitions and {dones} dones. Epoch {epoch}/{epochs}."
            )

        self.model.save(self.model_dir)
        self.close()
