import collections
import logging
from abc import abstractmethod

import mlflow
import numpy as np
from numpy.typing import NDArray

from corl.model.value_table import ModelValueTable
from corl.schemas.learning import Observation
from corl.schemas.tracker import StepKey, StepResult
from corl.trainer.trainer import Trainer
from corl.utils.config import Config

logger = logging.getLogger(__name__)


class TrainerMonteCarlo(Trainer):
    """
    First-visit Monte Carlo trainer for tabular Q-value models.

    Collects full episodes in batches, then applies first-visit Monte Carlo
    updates to a :class:`~corl.model.value_table.ModelValueTable`. Actions are
    selected with an ε-greedy policy whose exploration rate is decayed after
    each epoch.

    Attributes:
        gamma: Discount factor applied when computing episode returns.
        batch_size: Number of complete episodes to collect before updating
            the value table.
        epsilon: Current exploration rate for the ε-greedy policy.
        epsilon_decay: Multiplicative decay applied to ``epsilon`` after each
            epoch. Clipped to a minimum of ``0.01``.
        current_batch_done: Number of episodes completed in the current batch.
        current_batch_running: Number of episodes currently in progress.
        batch_worker_results: Maps worker ID → list of step results for the
            episode currently being collected by that worker.
        returns: Maps ``(observation_index, action)`` → sliding window of the
            last ``returns_window`` observed returns, used to compute the
            running mean Q-value estimate.
        returns_window: Maximum number of returns to keep per
            ``(observation_index, action)`` pair. Older returns are discarded
            as new ones arrive.
    """

    gamma: float
    batch_size: int
    epsilon: float
    epsilon_decay: float

    current_batch_done: int
    current_batch_running: int
    batch_worker_results: dict[int, list[StepResult]]
    returns: dict[tuple[int, int], collections.deque]

    def __init__(
        self,
        config: Config,
        model: ModelValueTable,
        model_checkpoint_frequency: int,
        observation_cardinality: int,
        observation_size: int,
        action_size: int,
        batch_size: int,
        gamma: float,
        epsilon: float,
        epsilon_decay: float,
        returns_window: int,
    ):
        """
        Initialise the Monte Carlo trainer.

        Args:
            config: Application configuration forwarded to the base
                :class:`~corl.trainer.trainer.Trainer`.
            model: Tabular Q-value model to train.
            model_checkpoint_frequency: Number of epochs between automatic
                weight checkpoints.
            observation_cardinality: Number of discrete bins per observation
                dimension (forwarded for reference; the model owns the table).
            observation_size: Number of observation dimensions.
            action_size: Number of discrete actions.
            batch_size: Number of complete episodes to collect per epoch.
            gamma: Discount factor ``γ ∈ (0, 1]``.
            epsilon: Initial exploration rate for the ε-greedy policy.
            epsilon_decay: Multiplicative decay applied to ``epsilon`` after
                each epoch, clipped to a minimum of ``0.01``.
            returns_window: Maximum number of returns to retain per
                ``(state, action)`` pair. Older returns are evicted
                automatically.
        """
        super().__init__(
            model=model,
            config=config,
            model_checkpoint_frequency=model_checkpoint_frequency,
        )
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.returns_window = returns_window

        self.current_batch_running = 0
        self.current_batch_done = 0
        self.batch_worker_results = {}
        self.returns = {}

    def params(self) -> dict[str, str | int | float]:
        """
        Return hyperparameters for MLflow logging.

        Merges Monte Carlo-specific hyperparameters with the model's own
        metadata (e.g. table dimensions).

        Returns:
            Dict of parameter name → value, suitable for
            ``mlflow.log_params``.
        """
        return {
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "epsilon_initial": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "returns_window": self.returns_window,
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

        Implementations should also filter out observations belonging to
        workers that are no longer active, to avoid feeding stale data into
        the value table update.

        Args:
            observations: List of ``(StepKey, Observation)`` pairs received
                from the environment.

        Returns:
            List of ``(StepKey, NDArray[np.float32])`` pairs containing the
            numerical representation of each observation.
        """

    def update_value_table(self, step_results: list[StepResult]) -> float:
        """
        Apply a first-visit Monte Carlo update to the value table.

        Iterates over the episode in reverse to compute discounted returns,
        updating only the first visit to each ``(state, action)`` pair.
        The Q-value is set to the running mean of all observed returns for
        that pair across all episodes.

        Args:
            step_results: Ordered list of step results for a single episode.

        Returns:
            The discounted return ``G`` from the first step of the episode.
        """
        logger.debug(f"Updating value table with {len(step_results)} step results.")
        g = 0.0
        visited = set()
        for step_result in reversed(step_results):
            g = step_result.reward + self.gamma * g
            observation_index = self.model.observation_to_index(step_result.observation)
            table_key = (observation_index, step_result.action)
            if table_key not in visited:
                visited.add(table_key)
                if table_key not in self.returns:
                    self.returns[table_key] = collections.deque(
                        maxlen=self.returns_window
                    )
                self.returns[table_key].append(g)
                self.model.value_table[observation_index][step_result.action] = np.mean(
                    self.returns[table_key]
                )
        return g

    def run(self, epochs: int) -> None:
        """
        Run the full Monte Carlo training loop.

        For each epoch:

        1. Log the current epsilon and collect a batch of ``batch_size``
           complete episodes via :meth:`run_batch`.
        2. Apply first-visit Monte Carlo updates to the value table via
           :meth:`update_value_table`.
        3. Log ``return_avg``, ``reward_avg``, ``episode_length_avg``,
           ``value_table_nonzero``, and ``epsilon`` to MLflow.
        4. Save a weight checkpoint every ``model_checkpoint_frequency`` epochs.
        5. Decay ``epsilon`` by ``epsilon_decay`` (floored at ``0.01``).

        After all epochs, the final model is saved and the trainer is closed.

        Args:
            epochs: Number of training epochs to run. Must be >= 1.

        Raises:
            ValueError: If ``epochs < 1``.
        """
        if epochs < 1:
            raise ValueError(f"Number of epochs must be >= 1, got {epochs}")

        mlflow.log_params(self.params())

        for epoch in range(epochs):
            logger.info(
                f"Starting epoch {epoch + 1}/{epochs} with epsilon {self.epsilon:.4f}"
            )
            batch_results = self.run_batch()
            episode_returns = []
            for episode in batch_results:
                episode_returns.append(self.update_value_table(episode))

            all_rewards = [
                step_result.reward
                for episode in batch_results
                for step_result in episode
            ]
            mlflow.log_metrics(
                {
                    "return_avg": float(np.mean(episode_returns)),
                    "reward_avg": float(np.mean(all_rewards)),
                    "episode_length_avg": float(
                        np.mean([len(episode) for episode in batch_results])
                    ),
                    "value_table_nonzero": int(
                        np.count_nonzero(self.model.value_table)
                    ),
                    "epsilon": self.epsilon,
                },
                step=epoch,
            )

            if (epoch + 1) % self.model_checkpoint_frequency == 0:
                self.model.save_weights(self.model_dir, checkpoint=True)

            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

        self.model.save(self.model_dir)
        self.close()

    def assign_workers(self, active_worker_ids: list[int]) -> None:
        """
        Register newly available workers for the current batch.

        For each active worker not already tracked, allocates a result buffer
        and increments :attr:`current_batch_running`, up to the batch size
        limit.

        Args:
            active_worker_ids: IDs of all workers currently reported as active
                by the tracker.
        """
        for worker_id in active_worker_ids:
            if (
                self.current_batch_running < self.batch_size
                and worker_id not in self.batch_worker_results
            ):
                self.batch_worker_results[worker_id] = []
                self.current_batch_running += 1
                logger.info(
                    f"Worker {worker_id} started episode {self.current_batch_running}/{self.batch_size}."
                )

    def remove_inactive_workers(self, active_worker_ids: list[int]) -> None:
        """
        Discard buffered results for workers that have gone inactive.

        Any worker present in :attr:`batch_worker_results` but absent from
        ``active_worker_ids`` is removed and :attr:`current_batch_running` is
        decremented, so its slot can be filled by a new worker.

        Args:
            active_worker_ids: IDs of all workers currently reported as active
                by the tracker.
        """
        dead_workers = [
            worker_id
            for worker_id in self.batch_worker_results
            if worker_id not in active_worker_ids
        ]

        for worker_id in dead_workers:
            logger.info(
                f"Worker {worker_id} disappeared during batch. Removing from results."
            )
            del self.batch_worker_results[worker_id]
            self.current_batch_running -= 1

    def run_batch(self) -> list[list[StepResult]]:
        """
        Collect a batch of ``batch_size`` complete episodes.

        Loops until ``batch_size`` episodes have finished:

        1. Assigns newly available workers via :meth:`assign_workers`.
        2. Executes one training step via
           :meth:`~corl.trainer.trainer.Trainer.training_step`.
        3. Appends step results to each worker's buffer; finalises the buffer
           as a complete episode when ``step_result.done`` is ``True``.
        4. Removes workers that disappeared mid-batch via
           :meth:`remove_inactive_workers`.

        Returns:
            List of episodes, where each episode is an ordered list of
            :class:`~corl.schemas.tracker.StepResult` objects.
        """
        self.current_batch_running = 0
        self.current_batch_done = 0
        episodes = []

        while self.current_batch_done < self.batch_size:
            active_workers = self.tracker.worker_step_keys()
            active_worker_ids = [worker.worker_id for worker in active_workers]

            self.assign_workers(active_worker_ids)

            steps = self.training_step()

            for step_key, step_result in steps:
                self.batch_worker_results[step_key.worker_id].append(step_result)

                if step_result.done:
                    self.current_batch_done += 1
                    self.current_batch_running -= 1
                    episodes.append(self.batch_worker_results[step_key.worker_id])
                    del self.batch_worker_results[step_key.worker_id]
                    logger.info(
                        f"Worker {step_key.worker_id} finished episode {self.current_batch_done}/{self.batch_size}."
                    )

            self.remove_inactive_workers(active_worker_ids)

        return episodes
