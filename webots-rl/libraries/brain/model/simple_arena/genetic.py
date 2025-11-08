import random
import string

from brain.model.genetic import Genetic


class SimpleArenaGenetic(Genetic):
    """
    Genetic algorithm implementation for the Simple Arena environment.

    Inherits from:
        Genetic
    """

    def create_individual(self) -> list[float]:
        """
        Create a new individual for the population.

        Returns:
            list[float]: A list of random integers (0, 1, or 2) of length individual_size.
        """
        return [random.randint(0, 2) for _ in range(self.individual_size)]

    @staticmethod
    def crossover(parent_a: list[float], parent_b: list[float]) -> list[float]:
        """
        Perform single-point crossover between two parents.

        Args:
            parent_a (list[float]): The first parent individual.
            parent_b (list[float]): The second parent individual.

        Returns:
            list[float]: The child individual created by combining genes from both parents.
        """
        index = random.randint(1, len(parent_a) - 1)
        return parent_a[:index] + parent_b[index:]

    def mutate(self, individual) -> list[float]:
        """
        Mutate an individual by randomly changing its genes.

        Args:
            individual (list[float]): The individual to mutate.

        Returns:
            list[float]: The mutated individual.
        """
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = random.randint(0, 2)
        return individual

    @staticmethod
    def get_name() -> str:
        """
        Generate a unique name for the model by appending a random 4-character string.

        Returns:
            str: The generated model name, e.g., 'simple_arena_genetic_a1B2'.
        """
        rand_str = "".join(random.choices(string.ascii_letters + string.digits, k=4))
        return f"simple_arena_genetic_{rand_str}"
