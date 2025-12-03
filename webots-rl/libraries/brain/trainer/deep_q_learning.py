"""
Deep Q-Learning Trainer with Prioritized Experience Replay (PER).

This module provides a base trainer class for Deep Q-Learning algorithms that combines
Double DQN with Prioritized Experience Replay for efficient and stable training of
deep reinforcement learning agents.

Key Features:
    - **Double DQN**: Reduces Q-value overestimation by using separate networks for
      action selection and evaluation
    - **Prioritized Experience Replay (PER)**: Samples experiences based on their
      TD-error priorities, focusing learning on surprising/important transitions
    - **Importance Sampling**: Corrects bias introduced by non-uniform sampling with
      annealed importance sampling weights (beta: 0.4 → 1.0)
    - **Target Network**: Periodically updated frozen network for stable Q-value targets
    - **Epsilon-Greedy Exploration**: Decaying exploration strategy balancing exploration
      and exploitation
"""

import os
from abc import abstractmethod

import numpy as np
import tensorflow as tf
from brain.environment import Environment
from brain.model import MODEL_PATH
from brain.trainer import Trainer
from brain.utils.logger import logger
from brain.utils.prioritized_experience_replay import PrioritizedExperienceReplayBuffer

MODEL_SAVE_FREQUENCY = 500


class TrainerDeepQLearning(Trainer):
    """
    Deep Q-Learning trainer with Prioritized Experience Replay (PER) and Double DQN.

    This trainer implements a state-of-the-art Deep Q-Learning algorithm combining:
    - **Double DQN**: Uses a separate target network for stable Q-value targets, reducing overestimation bias.
    - **Prioritized Experience Replay (PER)**: Samples experiences based on their TD-error priorities,
      learning more from surprising/important transitions.
    - **Importance Sampling**: Corrects for bias introduced by non-uniform sampling using annealed weights.
    - **Epsilon-Greedy Exploration**: Balances exploration vs exploitation with decaying epsilon.

    Algorithm Overview:
    1. Store experiences in prioritized replay buffer with max priority for new experiences
    2. Sample batch proportionally to TD-error priorities
    3. Compute Q-values using main network, target Q-values using target network
    4. Calculate TD-errors and train with importance sampling weights
    5. Update experience priorities based on new TD-errors
    6. Periodically update target network weights
    7. Anneal epsilon (exploration) and beta (IS correction) over training

    Attributes:
        model (tf.keras.models.Sequential): Main Q-network trained at each step.
        target_model (tf.keras.models.Sequential): Target Q-network for stable value estimates,
            updated periodically from main model.
        memory (PrioritizedExperienceReplayBuffer): Replay buffer storing experiences with priorities.
        gamma (float): Discount factor for future rewards (typically 0.95-0.99).
        epsilon (float): Current exploration rate for epsilon-greedy policy.
        epsilon_decay (float): Multiplicative decay factor for epsilon per epoch (e.g., 0.995).
        batch_size (int): Number of experiences sampled per training step (typically 32-64).
        fit_frequency (int): Train the model every N environment steps.
        update_target_model_frequency (int): Copy main model weights to target model every N steps.
        episode_losses (list[float]): Loss values accumulated during current episode for logging.
        per_beta (float): Current importance sampling exponent (annealed from 0.4 to 1.0).
        per_beta_increment (float): Amount to increase beta per epoch for linear annealing.

    Key Methods:
        - fit_model(): Train Q-network on prioritized batch with IS weights
        - update_target_model(): Sync target network with main network
        - run(): Main training loop over epochs
        - simulation(): Abstract method to implement episode simulation (must be overridden)
    """

    model: tf.keras.models.Sequential | None
    target_model: tf.keras.models.Sequential | None
    memory: PrioritizedExperienceReplayBuffer
    gamma: float
    epsilon: float
    epsilon_decay: float
    batch_size: int
    fit_frequency: int
    update_target_model_frequency: int
    episode_losses: list[float]
    per_beta: float
    per_beta_increment: float

    def __init__(
        self,
        environment: Environment,
        model_name: str,
        model: tf.keras.models.Sequential,
        memory_size: int,
        gamma: float,
        epsilon: float,
        epsilon_decay: float,
        batch_size: int,
        fit_frequency: int,
        update_target_model_frequency: int,
        per_alpha: float,
        per_beta_start: float,
    ):
        """
        Initialize Deep Q-Learning trainer with Prioritized Experience Replay.

        Sets up the main Q-network, target Q-network, prioritized replay buffer, and all
        hyperparameters for training. The target network is initialized as a clone of the
        main network with identical weights.

        Args:
            environment (Environment): The training environment providing step() and reset() methods.
                Used to interact with the simulation and collect experiences.
            model_name (str): Base name for saving model checkpoints (e.g., "robot_navigation").
                The saved file will be "{model_name}.keras".
            model (tf.keras.models.Sequential): The Q-network model architecture.
                Should output Q-values for each action. Must be compiled with optimizer and loss.
            memory_size (int): Maximum capacity of the prioritized replay buffer.
                Typical values: 10,000 - 100,000. Larger buffers provide more diverse experiences
                but require more memory.
            gamma (float): Discount factor for future rewards in the Bellman equation.
                Range: [0, 1]. Typical values: 0.95 - 0.99.
                - Higher values (0.99): Agent prioritizes long-term rewards
                - Lower values (0.9): Agent focuses on immediate rewards
            epsilon (float): Initial exploration rate for epsilon-greedy policy.
                Range: [0, 1]. Typical start: 1.0 (100% random actions initially).
                Probability of taking a random action instead of the greedy action.
            epsilon_decay (float): Multiplicative decay factor applied to epsilon after each epoch.
                Range: (0, 1). Typical values: 0.995 - 0.999.
                Example: epsilon_decay=0.995 → epsilon decreases to ~0.01 after ~900 epochs.
            batch_size (int): Number of experiences sampled from replay buffer per training step.
                Typical values: 32, 64, 128.
                - Larger batches: More stable gradients but slower training
                - Smaller batches: Faster updates but noisier gradients
            fit_frequency (int): Train the Q-network every N environment steps.
                Typical values: 1 - 4.
                Example: fit_frequency=4 → train after every 4 steps in the environment.
            update_target_model_frequency (int): Copy main network weights to target network every N steps.
                Typical values: 500 - 10,000.
                - Lower values: Target network tracks main network closely (less stable)
                - Higher values: More stable learning but slower adaptation
            per_alpha (float): Priority exponent controlling how much prioritization is used.
                Range: [0, 1]. Typical value: 0.6.
                - alpha=0: Uniform sampling (no prioritization)
                - alpha=1: Full prioritization based on TD-errors
                - Intermediate values balance uniform and prioritized sampling
            per_beta_start (float): Initial importance sampling weight exponent for bias correction.
                Range: [0, 1]. Typical start: 0.4.
                Will be annealed to 1.0 over training epochs for full bias correction.
                - beta=0: No bias correction (faster learning early on)
                - beta=1: Full bias correction (unbiased updates for convergence)
        """
        super().__init__(environment=environment, model_name=model_name)
        self.model = model
        self.target_model = tf.keras.models.clone_model(model)
        self.memory = PrioritizedExperienceReplayBuffer(capacity=memory_size, alpha=per_alpha)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.fit_frequency = fit_frequency
        self.update_target_model_frequency = update_target_model_frequency
        self.episode_losses = []
        self.per_beta = per_beta_start
        self.per_beta_increment = 0

    def update_target_model(self) -> None:
        """
        Synchronize target network weights with the main Q-network.

        Copies all learned weights from the main Q-network (self.model) to the target network
        (self.target_model). This is a core component of Double DQN that stabilizes training
        by providing fixed Q-value targets for a period of time.

        Why This Is Important:
            In standard DQN, using the same network for both current Q-values and target Q-values
            creates a "moving target" problem where the target changes at every training step,
            leading to instability and divergence. By updating the target network only periodically,
            the targets remain stable between updates, allowing the main network to learn more
            effectively.

        When To Call:
            This method should be called every `update_target_model_frequency` steps during training.
            Typical frequencies: 500-10,000 steps.
            - Too frequent (e.g., every step): Defeats the purpose, reduces stability
            - Too rare (e.g., every 100k steps): Target network becomes too outdated
        """
        self.target_model.set_weights(self.model.get_weights())
        logger().debug("Target model weights updated from training model")

    def fit_model(self) -> None:
        """
        Train the Q-network on a prioritized batch of experiences.

        Implements the core training step combining Double DQN, Prioritized Experience Replay (PER),
        and Importance Sampling (IS) for bias correction. This method samples high-priority experiences,
        computes stable Q-value targets using the target network, trains the model with IS weights,
        and updates priorities based on new TD-errors.

        Algorithm Steps:
            1. **Early Return**: Skip training if replay buffer has fewer than batch_size experiences
            2. **Prioritized Sampling**: Sample batch_size experiences from PER buffer
               - Experiences with higher TD-errors are sampled more frequently
               - Returns states, actions, rewards, next_states, terminals, tree indices, and IS weights
            3. **Current Q-Values**: Predict Q-values for sampled states using main network
            4. **Target Q-Values**: Predict Q-values for next states using target network (Double DQN)
            5. **Bellman Update**: Compute target Q-values using Bellman equation:
               - Terminal states: target = reward (no future value)
               - Non-terminal states: target = reward + gamma * max(Q_target(s', a'))
            6. **TD-Error Computation**: Calculate prediction errors before training (for priority updates)
            7. **Train with IS Weights**: Fit model using importance sampling weights to correct bias
            8. **Priority Update**: Update experience priorities in PER buffer based on new TD-errors

        Double DQN Details:
            Uses two networks to reduce overestimation bias:
            - Main network (self.model): Continuously updated during training
            - Target network (self.target_model): Provides stable targets, updated periodically
            Target Q-value = reward + gamma * max(Q_target(next_state))

        Prioritized Experience Replay:
            - High TD-error experiences are sampled more frequently (learn from mistakes)
            - New experiences start with maximum priority (ensure they're seen at least once)
            - Priorities decay naturally as TD-errors decrease with learning

        Importance Sampling Weights:
            - Corrects bias from non-uniform sampling
            - Weights scale the gradient updates during backpropagation
            - Higher priority → lower IS weight (downscale over-represented samples)
            - Lower priority → higher IS weight (upscale under-represented samples)

        When To Call:
            Called every `fit_frequency` steps during simulation (e.g., every 4 steps).
            Should NOT be called until replay buffer contains at least batch_size experiences.
        """

        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, terminals, idxs, weights = self.memory.sample(
            self.batch_size, self.per_beta
        )

        logger().debug(
            f"states shape: {states.shape} next_states shape: {next_states.shape} "
            f"actions shape: {actions.shape} rewards shape: {rewards.shape} terminals shape: {terminals.shape}"
        )

        q_values = self.model.predict(states, verbose=0)

        next_q_values = self.target_model.predict(next_states, verbose=0)
        max_next_q = np.max(next_q_values, axis=1)

        target_q = q_values.copy()
        target_q[np.arange(self.batch_size), actions] = rewards
        non_terminal = ~terminals
        target_q[non_terminal, actions[non_terminal]] = rewards[non_terminal] + self.gamma * max_next_q[non_terminal]

        td_errors = target_q[np.arange(self.batch_size), actions] - q_values[np.arange(self.batch_size), actions]

        history = self.model.fit(states, target_q, epochs=1, verbose=0, sample_weight=weights)
        if history.history.get("loss"):
            self.episode_losses.append(history.history["loss"][0])

        self.memory.update_priorities(idxs, td_errors)

    def run(self, epochs: int) -> None:
        """
        Execute the main training loop for the specified number of epochs.

        Runs the complete Deep Q-Learning training process with Prioritized Experience Replay.
        Each epoch represents one full episode in the environment. The method handles epsilon
        annealing (exploration decay), beta annealing (importance sampling correction), metrics
        logging to TensorBoard, and periodic model checkpointing.

        Training Flow Per Epoch:
            1. Decay epsilon (exploration rate) multiplicatively
            2. Increment beta (importance sampling weight) linearly
            3. Run one complete episode via simulation()
            4. Calculate average loss for the episode
            5. Log metrics to TensorBoard (reward, epsilon, loss, beta)
            6. Reset environment for next episode
            7. Save model checkpoint every MODEL_SAVE_FREQUENCY epochs

        Epsilon Annealing:
            - Starts at initial epsilon (typically 1.0)
            - Multiplied by epsilon_decay each epoch (e.g., 0.995)
            - Clamped to minimum of 0.01 (always maintain 1% exploration)
            - Controls exploration vs exploitation trade-off

        Beta Annealing:
            - Starts at per_beta_start (typically 0.4)
            - Linearly increases to 1.0 over all epochs
            - Controls strength of importance sampling bias correction
            - Full correction (beta=1.0) ensures unbiased convergence

        Metrics Logged to TensorBoard:
            - DeepQLearning/Reward: Total reward achieved in the episode
            - DeepQLearning/Epsilon: Current exploration rate
            - DeepQLearning/Loss: Average training loss for the episode
            - DeepQLearning/PER_Beta: Current importance sampling exponent

        Model Checkpointing:
            - Saved every MODEL_SAVE_FREQUENCY epochs (default: 500)
            - Saves full Keras model to: MODEL_PATH/{model_name}.keras
            - Includes model architecture, weights, and optimizer state

        Args:
            epochs (int): Total number of training epochs (episodes) to run.
                Typical values: 1000-5000 depending on task complexity.
        """
        self.per_beta_increment = (1.0 - self.per_beta) / epochs

        for epoch in range(epochs):
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
            self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)
            reward = self.simulation()
            loss_avg = sum(self.episode_losses) / len(self.episode_losses) if self.episode_losses else 0.0
            self.tb_writer.add_scalar("DeepQLearning/Reward", reward, epoch)
            self.tb_writer.add_scalar("DeepQLearning/Epsilon", self.epsilon, epoch)
            self.tb_writer.add_scalar("DeepQLearning/Loss", loss_avg, epoch)
            self.tb_writer.add_scalar("DeepQLearning/PER_Beta", self.per_beta, epoch)
            self.episode_losses = []
            self.environment.reset()
            logger().info(
                f"Epoch {epoch + 1}/{epochs} completed with " f"epsilon {self.epsilon:.4f} and reward {reward}"
            )
            if epoch % MODEL_SAVE_FREQUENCY == 0:
                self.save_model()

        self.close_tb()

    def save_model(self) -> None:
        """
        Save the trained Q-network model to disk.

        Persists the complete Keras model including architecture, weights, optimizer state,
        and training configuration. This allows for later loading and continuation of training
        or deployment for inference.
        """
        path = os.path.join(MODEL_PATH, self.model_name + ".keras")
        self.model.save(path)
        logger().info(f"Model saved successfully at {path}")

    @abstractmethod
    def simulation(self) -> float:
        """
        Run one complete episode in the environment (abstract method).

        This method must be implemented by subclasses to define the specific episode logic
        for the training environment. It handles the interaction loop between the agent and
        environment, including observation collection, action selection, experience storage,
        model training, and target network updates.
        """
        raise NotImplementedError("Method simulation() not implemented in TrainerDoubleQLearning.")
