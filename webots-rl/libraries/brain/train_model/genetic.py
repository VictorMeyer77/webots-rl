import json
import os
import random
from abc import abstractmethod

from brain.environment import Environment
from brain.train_model import TrainModel
from brain.utils.logger import logger


class TrainGenetic(TrainModel):
    """
    Abstract base class for implementing genetic algorithms.

    Attributes:
        environment (Environment): The environment in which individuals are evaluated.
        generation_size (int): Number of individuals per generation.
        individual_size (int): Size of each individual (genome length).
        mutation_rate (float): Probability of mutating each gene.
        selection_rate (float): Fraction of top individuals selected for reproduction.
        epochs (int): Number of generations to run the algorithm.
        actions (list[float]): The best actions found by the algorithm.
    """

    generation_size: int
    individual_size: int
    mutation_rate: float
    selection_rate: float
    epochs: int
    actions: list[float]

    def __init__(
        self,
        environment: Environment,
        generation_size: int,
        individual_size: int,
        mutation_rate: float,
        selection_rate: float,
        epochs: int,
    ):
        """
        Initialize the genetic algorithm with the given parameters.

        Args:
            environment (EnvironmentGenetic): The environment for evaluation.
            generation_size (int): Number of individuals per generation.
            individual_size (int): Size of each individual.
            mutation_rate (float): Mutation probability per gene.
            selection_rate (float): Fraction of top individuals selected.
            epochs (int): Number of generations to run.
        """
        super().__init__()
        self.environment = environment
        self.generation_size = generation_size
        self.individual_size = individual_size
        self.mutation_rate = mutation_rate
        self.selection_rate = selection_rate
        self.epochs = epochs
        self.actions = []
        logger().info(
            f"Genetic Algorithm initialized with generation_size={generation_size}, "
            f"individual_size={individual_size}, mutation_rate={mutation_rate}, "
            f"selection_rate={selection_rate}, epochs={epochs}"
        )

    @abstractmethod
    def create_individual(self) -> list[float]:
        """
        Create a new individual (genome).
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Method create_individual() not implemented.")

    @abstractmethod
    def crossover(self, parent_a: list[float], parent_b: list[float]) -> list[float]:
        """
        Perform crossover between two parents to produce a child.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Method crossover() not implemented.")

    @abstractmethod
    def mutate(self, individual) -> list[float]:
        """
        Mutate an individual.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Method mutate() not implemented.")

    def evaluate_generation(self, population: list[list[float]]) -> list[tuple[list[float], float]]:
        """
        Evaluate the fitness of each individual in the population.

        Args:
            population (list[list[float]]): The current generation.

        Returns:
            list[tuple[list[float], float]]: List of (individual, reward) tuples.
        """
        fitness_scores = []
        for index, individual in enumerate(population):
            self.environment.reset()
            self.tcp_socket.send(json.dumps({"actions": individual}))
            reward = self.environment.run()
            logger().info(f"Individual {index} Reward: {reward}")
            fitness_scores.append(reward)
        return list(zip(population, fitness_scores))

    def run(self) -> tuple[list[float], float]:
        """
        Run the genetic algorithm for the specified number of epochs.

        Returns:
            tuple[list[float], float]: The best individual and its reward.
        """
        population = [self.create_individual() for _ in range(self.generation_size)]
        population_eval = []
        for epoch in range(self.epochs):
            population_eval = self.evaluate_generation(population)
            population_eval.sort(key=lambda x: x[1], reverse=True)
            population = [individual for individual, _ in population_eval]
            logger().info(f"EPOCH: {epoch}: Best Reward: {population_eval[0][1]}")
            next_gen = population[: int(self.generation_size * self.selection_rate) + 1]
            while len(next_gen) < self.generation_size:
                parent_a, parent_b = random.sample(next_gen, 2)
                child = self.crossover(parent_a, parent_b)
                child = self.mutate(child)
                next_gen.append(child)
            population = next_gen

        best_individual, best_reward = max(population_eval, key=lambda x: x[1])
        logger().info(f"Best Reward: {best_individual}")
        self.actions = best_individual
        return best_individual, best_reward

    def save(self):
        """
        Save the best actions found by the genetic algorithm to a JSON file.

        The file is saved in the model directory with the model's name as the filename.
        """
        model_path = os.path.join(self.model_dir, self.name + ".json")
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(self.actions, f)
        logger().info(f"Model saved successfully at {model_path}")
