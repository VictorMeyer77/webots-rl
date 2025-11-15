"""
Genetic training infrastructure for evolving action vectors using a simple
elitist genetic algorithm. Subclasses must implement genome creation,
crossover, and mutation strategies.
"""

import random
from abc import abstractmethod

import numpy as np
from brain.environment import Environment
from brain.model.genetic import ModelGenetic
from brain.train_model import Trainer
from brain.utils.logger import logger


class TrainerGenetic(Trainer):
    """
    Genetic algorithm trainer responsible for:
    - Managing population lifecycle across epochs.
    - Evaluating individuals via environment simulation.
    - Selecting, crossing, and mutating genomes.
    - Persisting the best performing individual.

    Attributes:
        generation_size: Number of individuals per generation.
        individual_size: Length of each genome (action vector).
        mutation_rate: Probability per gene to mutate.
        selection_rate: Fraction of top individuals retained for breeding.
        model: Holds the best actions for persistence.
    """

    generation_size: int
    individual_size: int
    mutation_rate: float
    selection_rate: float
    model: ModelGenetic

    def __init__(
        self,
        environment: Environment,
        model_name: str,
        generation_size: int,
        individual_size: int,
        mutation_rate: float,
        selection_rate: float,
    ):
        """
        Initialize the genetic trainer.

        Args:
            environment: Simulation environment instance.
            model_name: Identifier used for saving the model.
            generation_size: Size of each generation (population).
            individual_size: Genome length (number of actions).
            mutation_rate: Per-gene mutation probability in [0,1].
            selection_rate: Fraction (0–1] of elites retained each epoch.
        """
        super().__init__(environment=environment, model_name=model_name)
        self.generation_size = generation_size
        self.individual_size = individual_size
        self.mutation_rate = mutation_rate
        self.selection_rate = selection_rate
        self.model = ModelGenetic()

        logger().info(
            f"Genetic Algorithm initialized with generation_size={generation_size}, "
            f"individual_size={individual_size}, mutation_rate={mutation_rate}, "
            f"selection_rate={selection_rate}"
        )

    @abstractmethod
    def create_individual(self) -> np.ndarray:
        """
        Create a new genome.

        Returns:
            A NumPy array representing an individual's genome.
        """
        raise NotImplementedError("Method create_individual() not implemented.")

    @abstractmethod
    def crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        """
        Produce a child from two parents.

        Args:
            parent_a: First parent genome.
            parent_b: Second parent genome.

        Returns:
            Child genome resulting from crossover.
        """
        raise NotImplementedError("Method crossover() not implemented.")

    @abstractmethod
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Apply mutation to a genome.

        Args:
            individual: Genome to mutate (may be modified in-place).

        Returns:
            Mutated genome.
        """
        raise NotImplementedError("Method mutate() not implemented.")

    def simulation(self, actions: np.ndarray) -> float:
        """
        Run a full episode using provided actions.

        Communication phases:
            1. Sync handshake with controller.
            2. Send action vector once.
            3. Wait for controller step confirmation.
            4. Step environment; collect reward.
            5. Detect termination and exit loop.

        Args:
            actions: Action vector (genome) to evaluate.

        Returns:
            Mean reward over the episode.
        """
        rewards = []
        queue = self.environment.queue
        state = None
        sync = False
        send_actions = False
        step_control = False

        # Main supervisor-driven loop; exits on Webots termination (-1) or episode end.
        while self.environment.supervisor.step(self.environment.timestep) != -1:

            queue.clear_buffer()

            # (1) Initial synchronization handshake on the very first step.
            if not sync:
                if not queue.search_message("ack"):
                    queue.send({"sync": 1})
                    logger().debug("Sent sync message to controller.")
                    continue
                else:
                    sync = True
                    logger().debug("Synchronization with controller successful.")

            # (2) Actions dispatch to controller.
            if not send_actions:
                queue.send({"actions": actions.tolist()})
                send_actions = True

            # (3) Blocking wait for end step controller message.
            if not step_control:
                step_messages = queue.search_message("step")
                if not step_messages:
                    continue
                else:
                    step_object = step_messages[0]
                    if step_object["step"] != self.environment.step_index:
                        raise RuntimeError(
                            f"Controller step index {step_object['step']} does not match "
                            f"supervisor step index {self.environment.step_index}."
                        )
                    else:
                        step_control = True

            # (4) Environment step: obtain new state and reward.
            state, reward = self.environment.step()
            rewards.append(reward)
            logger().debug(
                f"Step {self.environment.step_index}: Distance to finish line: {state.finish_line_distance:.4f}"
            )

            # (5) Termination check: restart controller and exit loop if episode ends.
            if state.is_terminated:
                break

            self.environment.step_index += 1
            step_control = False

        logger().info(f"Simulation terminated at step {state.step_index}, success: {state.is_success}")
        return sum(rewards) / len(rewards)

    def evaluate_generation(self, population: list[np.ndarray]) -> list[tuple[np.ndarray, float]]:
        """Evaluate all individuals in the population.

        Args:
            population: List of genomes.

        Returns:
            List of (genome, fitness) tuples sorted externally later.
        """
        fitness_scores = []
        for index, individual in enumerate(population):
            reward = self.simulation(individual)
            logger().info(f"Individual {index} Reward: {reward}")
            fitness_scores.append(reward)
            self.environment.reset()
        return list(zip(population, fitness_scores))

    def run(self, epochs: int) -> None:
        """
        Execute the genetic optimization loop.

        Process per epoch:
            - Evaluate current population.
            - Sort by fitness (descending).
            - Retain top selection_rate fraction (elitism).
            - Fill remainder via crossover + mutation.
            - Track best reward via tensorboard.

        Args:
            epochs: Number of generations to evolve.
        """
        population = [self.create_individual() for _ in range(self.generation_size)]
        population_eval = []
        for epoch in range(epochs):
            population_eval = self.evaluate_generation(population)
            population_eval.sort(key=lambda x: x[1], reverse=True)
            population = [individual for individual, _ in population_eval]
            self.tb_writer.add_scalar("Genetic/Reward", population_eval[0][1], epoch)
            next_gen = population[: int(self.generation_size * self.selection_rate) + 1]
            while len(next_gen) < self.generation_size:
                parent_a, parent_b = random.sample(next_gen, 2)
                child = self.crossover(parent_a, parent_b)
                child = self.mutate(child)
                next_gen.append(child)
            population = next_gen

        best_individual, best_reward = max(population_eval, key=lambda x: x[1])
        self.model.actions = best_individual

    def save_model(self) -> None:
        """
        Persist the best model actions to storage.
        """
        self.model.save(self.model_name)
