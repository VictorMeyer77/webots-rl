import logging

import mlflow
from numpy.typing import NDArray
from corl.memory.prioritized_experience_replay import PrioritizedExperienceReplayBuffer
from corl.memory.transition import Transition as TransitionMemory

from corl.model.deep_value_table import ModelDeepValueTable
from corl.trainer.trainer import Trainer


from corl.utils.config import Config
import tensorflow as tf
import numpy as np


logger = logging.getLogger(__name__)


class TrainerDeepQLearning(Trainer):
    gamma: float
    epsilon: float
    epsilon_min: float
    epsilon_decay: float
    batch_size: int
    fit_frequency: int
    update_target_weights_frequency: int
    target_weights: tf.keras.Model
    per_beta: float
    per_beta_increment: float
    transition: TransitionMemory
    experience_replay: PrioritizedExperienceReplayBuffer

    def __init__(
        self,
        config: Config,
        model: ModelDeepValueTable,
        model_checkpoint_frequency: int,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        batch_size: int,
        fit_frequency: int,
        update_target_weights_frequency: int,
        per_size: int,
        per_alpha: float,
        per_beta_start: float,
    ):

        super().__init__(
            model=model,
            config=config,
            model_checkpoint_frequency=model_checkpoint_frequency,
        )
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.fit_frequency = fit_frequency
        self.update_target_weights_frequency = update_target_weights_frequency
        self.per_beta = per_beta_start
        self.transition = TransitionMemory()
        self.experience_replay = PrioritizedExperienceReplayBuffer(
            capacity=per_size, alpha=per_alpha
        )
        self.target_weights = tf.keras.models.clone_model(model.weights)
        self.target_weights.set_weights(model.weights.get_weights())

    def params(self) -> dict[str, str | int | float]:

        return {
            "gamma": self.gamma,
            "epsilon_initial": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "fit_frequency": self.fit_frequency,
            "update_target_weights_frequency": self.update_target_weights_frequency,
            "per_alpha": self.experience_replay.alpha,
            "per_beta": self.per_beta,
        } | self.model.metadata()

    def policy(self, observations: NDArray[np.float32]) -> NDArray[np.int32]:

        return self.model.epsilon_greedy_policy(observations, self.epsilon)

    def update_target_weights(self) -> None:

        self.target_weights.set_weights(self.model.weights.get_weights())
        logger.debug("Target model weights updated from training model")

    def fit_model(self) -> dict[str, float] | None:

        if len(self.experience_replay) < self.batch_size:
            return None

        observations, actions, rewards, next_observations, terminals, idxs, weights = (
            self.experience_replay.sample(self.batch_size, self.per_beta)
        )

        logger.debug(
            f"observations shape: {observations.shape} next_observations shape: {next_observations.shape} "
            f"actions shape: {actions.shape} rewards shape: {rewards.shape} terminals shape: {terminals.shape}"
        )

        next_q_values = self.target_weights(next_observations, training=False).numpy()
        max_next_q = np.max(next_q_values, axis=1)
        non_terminal = ~terminals

        current_q = self.model.weights(observations, training=False).numpy()
        td_targets = rewards + self.gamma * max_next_q * non_terminal

        target_q = current_q.copy()
        target_q[np.arange(self.batch_size), actions] = td_targets

        history = self.model.weights.fit(
            observations, target_q, epochs=1, verbose=0, sample_weight=weights
        )

        # TD errors use pre-fit Q values so no extra forward pass is needed
        td_errors = td_targets - current_q[np.arange(self.batch_size), actions]

        self.experience_replay.update_priorities(idxs, td_errors)

        return {
            "loss": history.history["loss"][0],
            "mean_td_error": float(np.mean(np.abs(td_errors))),
            "mean_reward": float(np.mean(rewards)),
            "mean_max_next_q": float(np.mean(max_next_q)),
        }

    def run(self, epochs: int) -> None:

        if epochs < 1:
            raise ValueError(f"Number of epochs must be >= 1, got {epochs}")

        mlflow.log_params(self.params())

        # epoch = 0
        training_step_count = 0
        last_fit = 0
        last_target_update = 0
        last_checkpoint = 0
        self.per_beta_increment = (1.0 - self.per_beta) / epochs

        logger.info(
            f"Starting training for {epochs} episodes, beta increment: {self.per_beta_increment}"
        )

        while training_step_count < epochs:
            steps = self.training_step()
            training_step_count += len(steps)

            transitions = [
                transition
                for step_key, step_result in steps
                for transition in self.transition.make(step_key.worker_id, step_result)
            ]

            for transition in transitions:
                self.experience_replay.add(
                    transition.current_step.observation,
                    transition.current_step.action,
                    transition.current_step.reward,
                    transition.next_step.observation
                    if transition.next_step
                    else transition.current_step.observation,
                    transition.current_step.done,
                )

            if training_step_count - last_fit >= self.fit_frequency:
                fit_metrics = self.fit_model()
                if fit_metrics:
                    mlflow.log_metrics(
                        fit_metrics
                        | {
                            "epsilon": self.epsilon,
                            "per_size": len(self.experience_replay),
                            "per_beta": self.per_beta,
                        },
                        step=training_step_count,
                    )
                last_fit = training_step_count

            if (
                training_step_count - last_target_update
                >= self.update_target_weights_frequency
            ):
                self.update_target_weights()
                last_target_update = training_step_count

            if training_step_count - last_checkpoint >= self.model_checkpoint_frequency:
                self.model.save_weights(self.model_dir, checkpoint=True)
                last_checkpoint = training_step_count

            for _ in transitions:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

            logger.debug(
                f"Processed {len(steps)} steps with {len(transitions)}. Epoch {training_step_count}/{epochs}."
            )

        self.model.save(self.model_dir)
        self.close()
