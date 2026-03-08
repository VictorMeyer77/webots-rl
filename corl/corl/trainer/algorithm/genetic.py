import logging
import random
from abc import abstractmethod
from collections import deque

import mlflow
import numpy as np
from numpy.typing import NDArray

from corl.model.genetic import ModelGenetic
from corl.schemas.learning import Observation
from corl.schemas.tracker import StepKey, StepResult
from corl.trainer.trainer import Trainer
from corl.utils.config import Config

logger = logging.getLogger(__name__)


class TrainerGenetic(Trainer):
    """
    Abstract base class for elitist genetic algorithm trainers.

    Manages the full evolutionary loop: genome initialisation, parallel
    worker evaluation, elitist selection, crossover, and mutation.
    Concrete subclasses must implement :meth:`create_individual`,
    :meth:`crossover`, and :meth:`mutate` to define the genome
    representation and genetic operators for a specific task.

    The training loop distributes individuals from a queue to available
    workers. Each worker runs a full episode and returns a cumulative
    reward. Once all individuals in a generation have been evaluated,
    the top ``selection_rate`` fraction are kept as elites and the
    remainder are filled by crossing over and mutating pairs drawn
    from the elite pool.

    Attributes:
        generation_size: Number of individuals per generation.
        individual_size: Length of each genome (action vector).
        mutation_rate: Probability of mutating each gene during :meth:`mutate`.
        selection_rate: Fraction of the top individuals carried over as
            elites into the next generation. At least 2 elites are always
            kept to guarantee a valid parent pool.
        generation_queue: FIFO queue of individuals waiting to be assigned
            to a worker for evaluation in the current generation.
        worker_generation_map: Maps each active worker ID to the individual
            it is currently evaluating.
        worker_reward_map: Accumulates the episode reward for each active
            worker. Reset to ``0.0`` when a new individual is assigned.
    """

    generation_size: int
    individual_size: int
    mutation_rate: float
    selection_rate: float

    generation_queue: deque[NDArray[np.int32]] = deque()
    worker_generation_map: dict[int, NDArray[np.int32]] = {}
    worker_reward_map: dict[int, float] = {}

    def __init__(
        self,
        config: Config,
        model_checkpoint_frequency: int,
        generation_size: int,
        individual_size: int,
        mutation_rate: float,
        selection_rate: float,
    ):
        """
        Initialise the genetic trainer.

        Delegates infrastructure setup (API, tracker, MLflow, TensorBoard)
        to the parent :class:`~corl.trainer.trainer.Trainer`, then stores
        the GA-specific hyper-parameters.

        Args:
            config: Application configuration containing ``trainer_output_dir``,
                ``api_host``, and ``api_port``.
            model_checkpoint_frequency: Number of epochs between automatic
                weight checkpoints.
            generation_size: Number of individuals to evaluate per generation.
            individual_size: Number of genes (actions) in each individual.
            mutation_rate: Per-gene probability of mutation applied by
                :meth:`mutate`.
            selection_rate: Fraction of top-ranked individuals selected as
                elites for the next generation. Clamped to a minimum of 2
                individuals.
        """
        super().__init__(
            model=ModelGenetic(),
            config=config,
            model_checkpoint_frequency=model_checkpoint_frequency,
        )
        self.generation_size = generation_size
        self.individual_size = individual_size
        self.mutation_rate = mutation_rate
        self.selection_rate = selection_rate

    @abstractmethod
    def create_individual(self) -> NDArray[np.int32]:
        """
        Generate a single random individual (genome).

        Returns:
            NDArray[np.int32]: A 1-D integer array of length
            :attr:`individual_size` whose values represent discrete actions.
        """
        raise NotImplementedError("Method create_individual() not implemented.")

    @abstractmethod
    def crossover(
        self, parent_a: NDArray[np.int32], parent_b: NDArray[np.int32]
    ) -> NDArray[np.int32]:
        """
        Produce a child genome by combining two parent genomes.

        Args:
            parent_a: First parent genome.
            parent_b: Second parent genome.

        Returns:
            NDArray[np.int32]: Child genome of the same length as the parents.
        """
        raise NotImplementedError("Method crossover() not implemented.")

    @abstractmethod
    def mutate(self, individual: NDArray[np.int32]) -> NDArray[np.int32]:
        """
        Apply random mutations to an individual genome.

        Each gene is independently mutated with probability
        :attr:`mutation_rate`.

        Args:
            individual: The genome to mutate.

        Returns:
            NDArray[np.int32]: Mutated genome of the same length as the input.
        """
        raise NotImplementedError("Method mutate() not implemented.")

    def params(self) -> dict[str, str | int | float]:
        """
        Return the GA hyper-parameters for MLflow logging.

        Returns:
            dict[str, str | int | float]: Dictionary with keys
            ``generation_size``, ``individual_size``, ``mutation_rate``,
            and ``selection_rate``.
        """
        return {
            "generation_size": self.generation_size,
            "individual_size": self.individual_size,
            "mutation_rate": self.mutation_rate,
            "selection_rate": self.selection_rate,
        }

    def policy(self, observations: NDArray[np.float32]) -> NDArray[np.int32]:
        """
        Resolve batch observations to pre-assigned genome actions.

        The genetic algorithm does not learn from observations directly.
        Instead, each observation encodes ``(worker_id, step)`` — a lookup
        key into :attr:`worker_generation_map` — so the correct action for
        the current step is returned without any inference.

        Args:
            observations: Array of shape ``(N, 2)`` where each row is
                ``[worker_id, step]`` as ``float32``.

        Returns:
            NDArray[np.int32]: Array of ``N`` discrete actions, one per
            observation.

        Raises:
            RuntimeError: If a worker ID in ``observations`` has no entry
                in :attr:`worker_generation_map`.
        """
        actions = []
        for observation in observations:
            worker_id = int(observation[0])
            step = int(observation[1])
            if worker_id not in self.worker_generation_map:
                raise RuntimeError(
                    f"Worker ID {worker_id} not found in generation map."
                )
            actions.append(self.worker_generation_map[worker_id][step])

        logger.debug(
            f"policy(): resolved {len(actions)} action(s) for workers "
            f"{[int(obs[0]) for obs in observations]}"
        )
        return np.array(actions, dtype=np.int32)

    def parse_observations(
        self, observations: list[tuple[StepKey, Observation]]
    ) -> list[tuple[StepKey, NDArray[np.float32]]]:
        """
        Convert raw step observations into ``(worker_id, step)`` lookup vectors.

        Filters out observations from workers that are not currently present
        in :attr:`worker_generation_map` (e.g. workers that finished their
        episode but whose ``done`` signal has not yet been processed).

        Args:
            observations: List of ``(StepKey, Observation)`` pairs received
                from the API in the current training step.

        Returns:
            list[tuple[StepKey, NDArray[np.float32]]]: Filtered list of
            ``(StepKey, array([worker_id, step]))`` pairs ready to be passed
            to :meth:`policy`.
        """
        return [
            (key, np.array([key.worker_id, key.step]).astype(np.float32))
            for key, _ in observations
            if key.worker_id in self.worker_generation_map
        ]

    def assign_generation_to_workers(self, active_worker_ids: list[int]) -> None:
        """
        Pop individuals from :attr:`generation_queue` and assign them to idle workers.

        A worker is considered idle when it is present in ``active_worker_ids``
        but absent from :attr:`worker_generation_map`. The reward accumulator
        for each newly assigned worker is reset to ``0.0``.

        Args:
            active_worker_ids: List of worker IDs currently reported as
                active by the tracker.
        """
        for worker_id in active_worker_ids:
            if (
                worker_id not in self.worker_generation_map
                and len(self.generation_queue) > 0
            ):
                individual = self.generation_queue.pop()
                self.worker_generation_map[worker_id] = individual
                self.worker_reward_map[worker_id] = 0.0
                logger.debug(f"Assigned new individual to worker {worker_id}")

    def process_step_results(
        self, steps: list[tuple[StepKey, StepResult]]
    ) -> list[tuple[NDArray[np.int32], float]]:
        """
        Accumulate rewards and collect completed episode evaluations.

        For each step result, the reward is added to the running total in
        :attr:`worker_reward_map`. When a step is marked ``done``, the
        ``(genome, total_reward)`` pair is recorded, and the worker's
        entries are removed from both maps so it can be reassigned.

        Args:
            steps: List of ``(StepKey, StepResult)`` pairs from the
                current training step.

        Returns:
            list[tuple[NDArray[np.int32], float]]: Completed
            ``(genome, cumulative_reward)`` pairs for all episodes that
            finished in this batch of steps.
        """
        evaluations = []

        for step_key, step_result in steps:
            worker_id = step_key.worker_id
            reward = step_result.reward

            if worker_id in self.worker_reward_map:
                self.worker_reward_map[worker_id] += reward

                if step_result.done:
                    evaluations.append(
                        (
                            self.worker_generation_map[worker_id],
                            self.worker_reward_map[worker_id],
                        )
                    )
                    logger.info(
                        f"Worker {worker_id} completed episode with reward {self.worker_reward_map[worker_id]:.2f}"
                    )
                    del self.worker_generation_map[worker_id]
                    del self.worker_reward_map[worker_id]

        return evaluations

    def remove_inactive_workers(self, active_worker_ids: list[int]) -> None:
        """
        Re-queue individuals from workers that have gone offline.

        Any worker present in :attr:`worker_generation_map` but absent from
        ``active_worker_ids`` is considered dead. Its individual is pushed
        back onto :attr:`generation_queue` so it will be re-evaluated by
        another worker. Accumulated reward for the dead worker is discarded.

        Args:
            active_worker_ids: List of worker IDs currently reported as
                active by the tracker.
        """
        dead_workers = [
            worker_id
            for worker_id in self.worker_generation_map.keys()
            if worker_id not in active_worker_ids
        ]
        for worker_id in dead_workers:
            logger.warning(
                f"Worker {worker_id} is no longer active. Removing from generation map."
            )
            self.generation_queue.append(self.worker_generation_map[worker_id])
            del self.worker_generation_map[worker_id]
            del self.worker_reward_map[worker_id]

    def evaluate_generation(self) -> list[tuple[NDArray[np.int32], float]]:
        """
        Run all individuals in the current generation and return ranked results.

        Drives the training loop until every individual in
        :attr:`generation_queue` has been evaluated: dispatches individuals
        to workers via :meth:`assign_generation_to_workers`, collects step
        results, and handles worker churn via :meth:`remove_inactive_workers`.

        Once all ``generation_size`` evaluations are complete, results are
        sorted in descending order by cumulative reward.

        Returns:
            list[tuple[NDArray[np.int32], float]]: All ``(genome, reward)``
            pairs for the generation, sorted best-first.

        Raises:
            RuntimeError: If :attr:`generation_queue` is empty when called.
        """
        if len(self.generation_queue) == 0:
            raise RuntimeError("Generation queue is empty. Cannot evaluate generation.")

        evaluations = []

        while len(evaluations) < self.generation_size:
            workers = self.tracker.worker_step_keys()
            worker_ids = [worker.worker_id for worker in workers]
            self.assign_generation_to_workers(worker_ids)
            steps = self.training_step()
            evaluations += self.process_step_results(steps)
            self.remove_inactive_workers(worker_ids)
            logger.debug(
                f"evaluate_generation(): {len(evaluations)}/{self.generation_size} individuals evaluated"
            )

        evaluations.sort(key=lambda x: x[1], reverse=True)

        return evaluations

    def run(self, epochs: int) -> None:
        """
        Execute the full evolutionary training loop.

        For each epoch:

        1. Evaluate the current generation via :meth:`evaluate_generation`.
        2. Log the best reward to MLflow.
        3. Save a weight checkpoint every ``model_checkpoint_frequency`` epochs.
        4. Evolve the next generation via :meth:`generate_next_generation`.

        After all epochs, the best individual's weights are saved and
        :meth:`~corl.trainer.trainer.Trainer.close` is called to finalise
        artefacts.

        Args:
            epochs: Number of generations to evolve.
        """
        mlflow.log_params(self.params())

        best_individual, best_reward = None, None
        self.generation_queue = deque(
            [self.create_individual() for _ in range(self.generation_size)]
        )

        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch + 1}/{epochs}")

            generation_eval = self.evaluate_generation()
            best_individual, best_reward = generation_eval[0]
            logger.info(f"Epoch {epoch + 1}/{epochs} — best reward: {best_reward:.2f}")
            mlflow.log_metrics({"reward": best_reward}, step=epoch)

            if epoch > 0 and epoch % self.model_checkpoint_frequency == 0:
                self.model.set_weights(best_individual)
                self.model.save_weights(self.model_dir, checkpoint=True)

            generation = [individual for individual, _ in generation_eval]
            next_generation = self.generate_next_generation(generation)

            self.generation_queue = deque(next_generation)

        logger.info(
            f"Best individual after {epochs} epochs: {best_individual} with reward {best_reward:.2f}"
        )

        self.model.set_weights(best_individual)
        self.model.save(self.model_dir)
        self.close()

    def generate_next_generation(
        self, current_generation: list[NDArray[np.int32]]
    ) -> list[NDArray[np.int32]]:
        """
        Build the next generation from the current ranked population.

        Keeps the top ``max(2, round(generation_size * selection_rate))``
        individuals as elites, then fills the remainder by randomly sampling
        two parents from the elite pool and applying :meth:`crossover`
        followed by :meth:`mutate`.

        Args:
            current_generation: Full population sorted best-first (as
                returned by :meth:`evaluate_generation`).

        Returns:
            list[NDArray[np.int32]]: New population of exactly
            ``generation_size`` individuals.
        """
        next_generation = current_generation[
            : max(2, round(self.generation_size * self.selection_rate))
        ]
        logger.debug(
            f"Selected top {len(next_generation)} individuals for next generation."
        )
        while len(next_generation) < self.generation_size:
            parent_a, parent_b = random.sample(next_generation, 2)
            child = self.crossover(parent_a, parent_b)
            child = self.mutate(child)
            next_generation.append(child)
        return next_generation
