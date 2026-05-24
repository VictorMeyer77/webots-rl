import collections
import json
import logging
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
from numpy.typing import NDArray

from corl.model.discrete.value_table import ModelValueTable
from corl.schemas.tracker import StepResult
from corl.trainer.trainer import Trainer
from corl.utils.config import Config

logger = logging.getLogger(__name__)


class TrainerMonteCarlo(Trainer):
    """
    First-visit Monte Carlo trainer for tabular Q-value models.

    Collects full episodes in batches, then applies first-visit Monte Carlo
    updates to a :class:`~corl.model.value_table.ModelValueTable`. Actions are
    selected with an ε-greedy policy whose exploration rate is decayed once
    per transition.

    Attributes:
        gamma: Discount factor applied when computing episode returns.
        batch_size: Number of complete episodes to collect per batch before
            updating the value table.
        epsilon: Current exploration rate for the ε-greedy policy.
        epsilon_min: Lower bound for epsilon; decay stops here.
        epsilon_decay: Multiplicative factor applied to ``epsilon`` once per
            transition. Clipped to a minimum of ``epsilon_min``.
        current_batch_done: Number of episodes completed in the current batch.
        current_batch_running: Number of episodes currently in progress.
        batch_count: Total number of batches completed since training started.
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
    epsilon_min: float
    epsilon_decay: float

    current_batch_done: int
    current_batch_running: int
    batch_count: int
    batch_worker_results: dict[int, list[StepResult]]
    returns: dict[tuple[int, int], collections.deque]
    returns_window: int

    def __init__(
        self,
        config: Config,
        model: ModelValueTable,
        checkpoint_frequency: int,
        batch_size: int,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        returns_window: int,
        checkpoint_id: str | None = None,
    ):
        """
        Initialise the Monte Carlo trainer.

        Args:
            config: Application configuration forwarded to the base
                :class:`~corl.trainer.trainer.Trainer`.
            model: Tabular Q-value model to train.
            checkpoint_frequency: Minimum number of transitions between
                automatic weight checkpoints.
            batch_size: Number of complete episodes to collect per batch.
            gamma: Discount factor ``γ ∈ (0, 1]``.
            epsilon: Initial exploration rate for the ε-greedy policy.
            epsilon_min: Lower bound for epsilon; decay stops once this value
                is reached.
            epsilon_decay: Multiplicative factor applied to ``epsilon`` once
                per transition, clipped to a minimum of ``epsilon_min``.
            returns_window: Maximum number of returns to retain per
                ``(state, action)`` pair. Older returns are evicted
                automatically.

        Raises:
            ValueError: If ``batch_size < 1``.
        """

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.returns_window = returns_window

        self.current_batch_running = 0
        self.current_batch_done = 0
        self.batch_count = 0
        self.batch_worker_results = {}
        self.returns = {}

        super().__init__(
            model=model,
            config=config,
            checkpoint_frequency=checkpoint_frequency,
            checkpoint_id=checkpoint_id,
        )

    def params(self) -> dict[str, str | int | float | bool]:
        """
        Return hyperparameters logged to MLflow at the start of training.

        Combines Monte Carlo-specific hyperparameters with model metadata so that every
        run is fully reproducible from the logged params alone.

        Returns:
            dict[str, str | int | float | bool]: Flat mapping of parameter names to
                their values, ready to pass to ``mlflow.log_params()``.
        """
        return super().params() | self.model.metadata()

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

    def update_value_table(self, step_results: list[StepResult]) -> tuple[float, float]:
        """
        Apply a first-visit Monte Carlo update to the value table.

        Iterates over the episode in reverse to compute discounted returns,
        updating only the first visit to each ``(state, action)`` pair.
        The Q-value is set to the running mean of all observed returns for
        that pair across all episodes.

        Args:
            step_results: Ordered list of step results for a single episode.

        Returns:
            Tuple of ``(G, reward_total)`` where ``G`` is the full discounted
            episode return at ``t=0`` (i.e. the return from the first step,
            computed as the episode accumulates backwards), and
            ``reward_total`` is the undiscounted sum of all rewards in the
            episode.
        """
        logger.debug(f"Updating value table with {len(step_results)} step results.")
        g = 0.0
        reward_total = 0.0
        visited = set()
        for step_result in reversed(step_results):
            g = step_result.reward + self.gamma * g
            reward_total += step_result.reward
            observation_index = self.model.observation_to_index(step_result.observation)
            action_idx = int(step_result.action[0])
            table_key = (observation_index, action_idx)
            if table_key not in visited:
                visited.add(table_key)
                if table_key not in self.returns:
                    self.returns[table_key] = collections.deque(
                        maxlen=self.returns_window
                    )
                self.returns[table_key].append(g)
                self.model.value_table[observation_index][action_idx] = np.mean(
                    self.returns[table_key]
                )
        return g, reward_total

    def checkpoint(self) -> None:
        """
        Persist all training state to prevent data loss on failure.

        Saves the model weights under ``<checkpoint_dir>/<timestamp>/model/``
        and all serialisable hyperparameters to ``<checkpoint_dir>/<timestamp>/params.json``.
        The timestamp-based subdirectory ensures successive checkpoints do not
        overwrite each other.
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
        restores declared class attributes from ``<checkpoint_dir>/<checkpoint_id>/params.json``.
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

        with open(path / "params.json", "r") as f:
            params = json.load(f)
            for key, value in params.items():
                if key in allowed:
                    setattr(self, key, value)

        logger.info(f"Recovered training state from {path}")

    def run(self, max_transitions: int) -> None:
        """
        Run the full Monte Carlo training loop.

        Iterates until ``max_transitions`` total transitions have been collected.
        Each iteration:

        1. Collects a batch of ``batch_size`` complete episodes via
           :meth:`run_batch`.
        2. Applies first-visit Monte Carlo updates to the value table via
           :meth:`update_value_table` for each episode in the batch.
        3. Logs the following metrics to MLflow once per completed batch:
           ``episode_return_avg``, ``episode_return_std``,
           ``episode_reward_avg``, ``episode``, ``batch_count``,
           ``transition_per_episode``, ``value_table_nonzero``, ``epsilon``.
        4. Saves a weight checkpoint every ``checkpoint_frequency``
           transitions.
        5. Decays ``epsilon`` by ``epsilon_decay`` once per transition in
           the batch (floored at ``epsilon_min``).

        After all transitions, the final model is saved and the trainer is
        closed.

        Args:
            max_transitions: Total number of environment transitions to collect before
                stopping. Must be >= 1.

        Raises:
            ValueError: If ``max_transitions < 1``.
        """
        if max_transitions < 1:
            raise ValueError(
                f"Number of max_transitions must be >= 1, got {max_transitions}"
            )

        if self.last_checkpoint_transition == 0:
            self._mlflow_log_train_params()
            training_step_count = 0
        else:
            training_step_count = self.last_checkpoint_transition

        while training_step_count < max_transitions:
            logger.info(
                f"Starting batch {self.batch_count + 1} with epsilon {self.epsilon:.4f}. Transitions {training_step_count} / {max_transitions}"
            )
            batch_results = self.run_batch()

            batch_transition_count = sum([len(episode) for episode in batch_results])
            training_step_count += batch_transition_count
            self.episode_count += len(batch_results)
            self.batch_count += 1

            episode_metrics = []
            for episode in batch_results:
                episode_metrics.append(self.update_value_table(episode))

            mlflow.log_metrics(
                {
                    "episode_return_avg": float(
                        np.mean([metric[0] for metric in episode_metrics])
                    ),
                    "episode_return_std": float(
                        np.std([metric[0] for metric in episode_metrics])
                    ),
                    "episode_reward_avg": float(
                        np.mean([metric[1] for metric in episode_metrics])
                    ),
                    "episode": self.episode_count,
                    "batch_count": self.batch_count,
                    "transition_per_episode": float(
                        np.mean([len(episode) for episode in batch_results])
                    ),
                    "value_table_nonzero": int(
                        np.count_nonzero(self.model.value_table)
                    ),
                    "epsilon": self.epsilon,
                },
                step=training_step_count,
            )

            if (
                training_step_count - self.last_checkpoint_transition
                >= self.checkpoint_frequency
            ):
                self.last_checkpoint_transition = training_step_count
                self.checkpoint()

            for _ in range(batch_transition_count):
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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
