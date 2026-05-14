import numpy as np
from numpy.typing import NDArray

from corl.trainer.algorithm.discrete.genetic import TrainerGenetic
from corl.utils.config import Config
from corl.utils.logger import setup_logging

MODEL_CHECKPOINT_FREQUENCY = 10  # Save model checkpoints every epochs
EPISODE_SIZE = 75  # Number of training steps per episode
MUTATION_RATE = 0.05  # Probability of mutation per gene
GENERATION_SIZE = 200  # Number of individuals in each generation
SELECTION_RATE = 0.1  # Proportion of individuals selected for reproduction
EPOCHS = 100  # Number of epochs to train
ACTION_SIZE = 9  # Number of possible actions


class SimpleArenaGenetic(TrainerGenetic):
    """
    Genetic algorithm trainer for the simple arena task.

    Evolves a population of fixed-length action sequences (genomes) to
    navigate the e-puck to the finish line. Each individual is an array
    of integer action indices replayed open-loop during evaluation.
    """

    def create_individual(self) -> NDArray[np.int32]:
        """
        Generate a random individual (genome).

        Returns:
            NDArray[np.int32]: Array of shape ``(individual_size,)`` with
            random action indices in ``[0, ACTION_SIZE)``.
        """
        return np.random.randint(
            0, ACTION_SIZE, size=self.individual_size, dtype=np.int32
        )

    def crossover(
        self, parent_a: NDArray[np.int32], parent_b: NDArray[np.int32]
    ) -> NDArray[np.int32]:
        """
        Produce an offspring via single-point crossover.

        A random cut point is chosen and the offspring takes genes from
        ``parent_a`` up to that point and from ``parent_b`` after it.

        Args:
            parent_a (NDArray[np.int32]): First parent genome.
            parent_b (NDArray[np.int32]): Second parent genome.

        Returns:
            NDArray[np.int32]: Child genome of the same length as the parents.
        """
        index = np.random.randint(1, parent_a.shape[0])
        return np.concatenate((parent_a[:index], parent_b[index:]))

    def mutate(self, individual: NDArray[np.int32]) -> NDArray[np.int32]:
        """
        Apply random mutation to an individual in-place.

        Each gene is independently replaced with a random action index with
        probability ``mutation_rate``.

        Args:
            individual (NDArray[np.int32]): Genome to mutate. Modified in-place.

        Returns:
            NDArray[np.int32]: The mutated genome (same object as input).
        """
        mutation_mask = np.random.rand(self.individual_size) < self.mutation_rate
        random_values = np.random.randint(
            0, ACTION_SIZE, size=self.individual_size, dtype=np.int32
        )
        individual[mutation_mask] = random_values[mutation_mask]
        return individual


if __name__ == "__main__":
    config = Config()
    setup_logging(config)

    SimpleArenaGenetic(
        config=config,
        model_checkpoint_frequency=MODEL_CHECKPOINT_FREQUENCY,
        generation_size=GENERATION_SIZE,
        individual_size=EPISODE_SIZE,
        mutation_rate=MUTATION_RATE,
        selection_rate=SELECTION_RATE,
    ).run(epochs=EPOCHS)
