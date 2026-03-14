import numpy as np
from corl.trainer.algorithm.genetic import TrainerGenetic

from corl.utils.config import Config
from corl.utils.logger import setup_logging
from numpy.typing import NDArray


MODEL_CHECKPOINT_FREQUENCY = 10
EPISODE_SIZE = 75
MUTATION_RATE = 0.05
GENERATION_SIZE = 40
SELECTION_RATE = 0.1
EPOCHS = 3



class SimpleArenaGenetic(TrainerGenetic):
    def create_individual(self) -> NDArray[np.int32]:
        return np.random.randint(0, 9, size=self.individual_size, dtype=np.int32)

    def crossover(
        self, parent_a: NDArray[np.int32], parent_b: NDArray[np.int32]
    ) -> NDArray[np.int32]:
        index = np.random.randint(1, parent_a.shape[0])
        return np.concatenate((parent_a[:index], parent_b[index:]))

    def mutate(self, individual: NDArray[np.int32]) -> NDArray[np.int32]:
        mutation_mask = np.random.rand(self.individual_size) < self.mutation_rate
        random_values = np.random.randint(
            0, 9, size=self.individual_size, dtype=np.int32
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
