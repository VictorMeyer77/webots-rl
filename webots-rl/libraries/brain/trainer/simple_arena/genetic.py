"""
Genetic trainer specialization for the Simple Arena environment.

This module provides a concrete implementation of a genetic algorithm
trainer for an environment where each genome is an integer action vector
with values in the discrete range [0, 2].
"""

import numpy as np
from brain.trainer.genetic import TrainerGenetic


class TrainerSimpleArenaGenetic(TrainerGenetic):
    """
    Concrete genetic algorithm trainer for the Simple Arena.

    Responsibilities:
      - Create new individuals (genomes).
      - Apply one-point crossover.
      - Apply per-gene mutation using a vectorized mask.

    Genome:
      A NumPy int64 array of length `individual_size` with each gene in {0,1,2}.
    """

    def create_individual(self) -> np.ndarray:
        """
        Create a new random genome.

        Returns:
            NumPy array of shape (individual_size,) with integer genes in {0,1,2,3}.
        """
        return np.random.randint(0, 4, size=self.individual_size, dtype=np.int64)

    @staticmethod
    def crossover(parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        """
        Perform one-point crossover between two parent genomes.

        The crossover index is sampled uniformly from [1, len(parent_a)-1].

        Args:
            parent_a: First parent genome (1D array).
            parent_b: Second parent genome (same shape as parent_a).

        Returns:
            Child genome composed of parent_a[:index] + parent_b[index:].

        Raises:
            ValueError: If parent shapes differ or are not 1D.
        """

        index = np.random.randint(1, parent_a.shape[0])
        return np.concatenate((parent_a[:index], parent_b[index:]))

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Mutate a genome in-place using a per-gene probability.

        A boolean mask selects genes to replace with new random values in {0,1,2}.

        Args:
            individual: Genome to mutate (modified in-place).

        Returns:
            The mutated genome (same array instance).
        """
        mask = np.random.rand(individual.shape[0]) < self.mutation_rate
        if np.any(mask):
            individual[mask] = np.random.randint(0, 3, size=mask.sum(), dtype=individual.dtype)
        return individual
