"""
REINFORCE Policy Gradient Trainer for Reinforcement Learning.

This module implements the REINFORCE algorithm (Williams, 1992), a Monte Carlo policy
gradient method that learns by directly optimizing the policy network to maximize
expected returns. REINFORCE uses complete episode trajectories to compute unbiased
gradient estimates of the policy objective.

Algorithm Overview:
    REINFORCE optimizes the policy π_θ(a|s) by following the policy gradient:

    ∇J(θ) = E[∇log π_θ(a_t|s_t) * G_t]

    Where:
    - θ: Policy network parameters
    - G_t: Discounted return from timestep t (Monte Carlo estimate)
    - π_θ(a|s): Probability of action a given state s under policy θ

Key Features:
    - **Monte Carlo Returns**: Uses full episode trajectories to compute G_t
    - **Entropy Regularization**: Adds entropy bonus to encourage exploration
    - **Return Normalization**: Standardizes returns per batch for stable gradients
    - **Gradient Clipping**: Prevents exploding gradients (clips by global norm)
    - **Batch Training**: Accumulates multiple episodes before updating policy

Mathematical Formulation:
    Policy Loss:
        L_policy = -E[log π_θ(a_t|s_t) * G_t]

    Entropy Bonus:
        L_entropy = -β * E[H[π_θ(·|s_t)]]
        where H[π] = -Σ π(a|s) log π(a|s)

    Total Loss:
        L_total = L_policy + L_entropy

    Discounted Return:
        G_t = Σ_{k=0}^{T-t} γ^k * r_{t+k}

Hyperparameters:
    - gamma (γ): Discount factor for future rewards [0.9-0.99]
    - episodes_per_batch: Number of episodes to accumulate before update [5-20]
    - entropy_beta (β): Entropy regularization coefficient [0.001-0.1]
    - normalize_returns: Whether to standardize returns per batch [True/False]
    - gradient_clip_norm: Maximum gradient norm for clipping [0.5-5.0]

"""

import os
from abc import abstractmethod

import numpy as np
import tensorflow as tf
from brain.environment import Environment
from brain.model import MODEL_PATH
from brain.trainer import Trainer
from brain.utils.logger import logger

MODEL_SAVE_FREQUENCY = 500  # Save model checkpoint every N epochs
GRADIENT_CLIP_NORM = 1.0  # Maximum gradient norm to prevent exploding gradients


class TrainerReinforce(Trainer):
    """
    REINFORCE trainer with entropy regularization and gradient clipping.

    Implements the REINFORCE algorithm (Monte Carlo policy gradient) with several
    enhancements for stability and exploration. This trainer accumulates multiple
    episodes into batches, computes discounted returns, and updates the policy
    network to maximize expected returns while maintaining exploration via entropy
    regularization.

    The trainer operates in an episodic manner:
    1. Run simulation() to collect one complete episode (observations, actions, rewards)
    2. Compute discounted returns G_t for the episode
    3. Accumulate into batch (repeat for episodes_per_batch episodes)
    4. Normalize returns across the batch
    5. Update policy network via gradient descent
    6. Repeat until convergence or max epochs

    Attributes:
        model (tf.keras.models.Model): Policy network that maps observations to action logits.
            Input: observation vector (state representation)
            Output: logits for each action (unnormalized log-probabilities)
        gamma (float): Discount factor for computing returns, range [0, 1].
            Higher values (0.99) emphasize long-term rewards.
            Lower values (0.9) focus on immediate rewards.
        episodes_per_batch (int): Number of episodes to accumulate before each policy update.
            Larger batches provide more stable gradients but slower updates.
            Typical values: 5-20 for simple tasks, 10-50 for complex tasks.
        entropy_beta (float): Coefficient for entropy regularization bonus.
            Higher values encourage more exploration (policy stays more random).
            Lower values allow faster convergence to deterministic policy.
            Typical range: 0.001-0.1 depending on action space and reward scale.
        normalize_returns (bool): Whether to standardize returns per batch.
            True: returns = (returns - mean) / (std + 1e-8)
            Helps with gradient stability but can hide signal if all episodes are similar.
    """

    model: tf.keras.models.Model | None
    gamma: float
    episodes_per_batch: int
    entropy_beta: float
    normalize_returns: bool = True

    def __init__(
        self,
        environment: Environment,
        model_name: str,
        model: tf.keras.models.Model,
        gamma: float,
        episodes_per_batch: int,
        entropy_beta: float,
        normalize_returns: bool = True,
    ):
        """
        Initialize the REINFORCE trainer with policy network and hyperparameters.

        Sets up the policy gradient trainer with all necessary components for training.
        The policy network must be pre-compiled with an optimizer (e.g., Adam) before
        being passed to this constructor.

        Args:
            environment (Environment): Training environment providing step() and reset() methods.
                Must implement the Environment interface with observation space and reward function.
            model_name (str): Base name for saving model checkpoints.
                Saved as "{model_name}.keras" in MODEL_PATH directory.
                Example: "robot_navigation" → "robot_navigation.keras"
            model (tf.keras.models.Model): Policy network (must be compiled with optimizer).
                Architecture:
                - Input: observation vector (e.g., sensor readings, image features)
                - Output: logits for each action (unnormalized log-probabilities)
                - Must NOT have softmax activation on output (logits are raw)
                Example:
                    model = Sequential([
                        Dense(64, activation='relu', input_shape=(obs_dim,)),
                        Dense(64, activation='relu'),
                        Dense(action_dim, activation='linear')
                    ])
                    model.compile(optimizer=Adam(learning_rate=0.001))
            gamma (float): Discount factor for future rewards, range [0, 1].
                Controls the trade-off between immediate and long-term rewards.
                - gamma = 0.9: Emphasizes near-term rewards (myopic)
                - gamma = 0.99: Values long-term rewards highly (far-sighted)
                - gamma = 0.999: Very long-term planning (can be unstable)
                Typical values: 0.95-0.99 for most tasks.
            episodes_per_batch (int): Number of episodes to collect before each policy update.
                Larger batches:
                - Pros: More stable gradients, better variance reduction
                - Cons: Slower updates, more memory usage
                Smaller batches:
                - Pros: Faster iteration, less memory
                - Cons: Noisier gradients, higher variance
                Typical values: 5-20 for simple tasks, 10-50 for complex environments.
            entropy_beta (float): Entropy regularization coefficient, range [0, ∞).
                Controls exploration vs exploitation trade-off.
                - Higher values (0.05-0.1): Strong exploration, policy stays stochastic
                - Medium values (0.01-0.02): Balanced exploration
                - Lower values (0.001-0.005): Minimal exploration, faster convergence
                - Zero: No entropy bonus, pure exploitation (not recommended)
                Recommendation: Start with 0.01 and adjust based on Mean_Entropy curves.
            normalize_returns (bool, optional): Whether to standardize returns per batch.
                Defaults to True.
                - True: Applies (returns - mean) / (std + 1e-8) normalization
                  - Pros: Stable gradient magnitudes, works across different reward scales
                  - Cons: Can hide signal if all episodes have similar returns
                - False: Uses raw returns
                  - Pros: Preserves reward signal magnitude
                  - Cons: Sensitive to reward scale, may need careful learning rate tuning
        """
        super().__init__(environment=environment, model_name=model_name)
        self.model = model
        self.gamma = gamma
        self.episodes_per_batch = episodes_per_batch
        self.entropy_beta = entropy_beta
        self.normalize_returns = normalize_returns

    def fit_model(self, observations: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> dict[str, float]:
        """
        Perform one policy gradient update on a batch of experiences.

        Implements the core REINFORCE update with entropy regularization and gradient clipping.
        This method computes the policy gradient ∇log π(a|s) * G_t, adds an entropy bonus
        for exploration, clips gradients to prevent instability, and applies the update to
        the policy network parameters.

        Algorithm Steps:
            1. **Normalize returns** (if enabled): Standardize to zero mean, unit variance
            2. **Forward pass**: Compute action logits and log-probabilities
            3. **Gather selected actions**: Extract log π(a_t|s_t) for actions actually taken
            4. **Policy loss**: Compute -E[log π(a_t|s_t) * G_t] (negative for gradient ascent)
            5. **Entropy bonus**: Compute -β * E[H[π(·|s_t)]] to encourage exploration
            6. **Total loss**: Sum policy loss and entropy loss
            7. **Gradient computation**: Compute ∇_θ L with respect to all trainable parameters
            8. **Gradient clipping**: Clip by global norm to prevent exploding gradients
            9. **Parameter update**: Apply clipped gradients via optimizer

        Args:
            observations (np.ndarray): Batch of state observations from collected episodes.
                Shape: (batch_size, observation_dim)
                - batch_size: Total number of timesteps across all episodes in the batch
                - observation_dim: Dimensionality of state representation
                Example: For 10 episodes of length ~50, batch_size ≈ 500
            actions (np.ndarray): Batch of actions taken at each timestep.
                Shape: (batch_size,)
                Values: Integer action indices in range [0, action_dim - 1]
                Must correspond 1-to-1 with observations (same order)
            returns (np.ndarray): Batch of discounted Monte Carlo returns G_t.
                Shape: (batch_size,)
                Values: G_t = Σ_{k=0}^{T-t} γ^k * r_{t+k} for each timestep
                Computed by discount_returns() before being passed here

        Returns:
            dict[str, float]: Dictionary of training metrics for logging/monitoring:
                - "loss" (float): Total loss (policy_loss + entropy_loss)
                - "policy_loss" (float): Policy gradient loss -E[log π(a_t|s_t) * G_t]
                - "entropy_loss" (float): Entropy regularization term -β * E[H[π]]
                - "grad_norm" (float): Global L2 norm of gradients before clipping
                - "mean_entropy" (float): Average entropy H[π] across batch

        Mathematical Details:
            Policy Gradient:
                ∇_θ J(θ) = E[∇_θ log π_θ(a_t|s_t) * G_t]
                Implemented as: policy_loss = -mean(log π(a_t|s_t) * G_t)

            Entropy Regularization:
                H[π(·|s)] = -Σ_a π(a|s) log π(a|s)
                entropy_loss = -entropy_beta * mean(H[π(·|s_t)])
                The negative sign encourages maximizing entropy (more exploration)

            Gradient Clipping:
                global_norm = sqrt(Σ ||∇_i||²)
                if global_norm > clip_norm:
                    ∇_i ← ∇_i * (clip_norm / global_norm)
                Prevents exploding gradients while preserving direction

        Side Effects:
            - Updates self.model's trainable parameters (weights and biases)
            - Advances optimizer state (e.g., Adam momentum terms)

        Notes:
            - Return normalization can cause issues if all returns are identical (std ≈ 0)
            - High gradient norms (> GRADIENT_CLIP_NORM) indicate potential instability
            - Low mean entropy (< 0.1 for discrete actions) means policy is very deterministic
            - Monitor these metrics in TensorBoard to diagnose training issues
        """
        if self.normalize_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        with tf.GradientTape() as tape:
            logits = self.model(observations)
            logits_prob = tf.nn.log_softmax(logits)

            action_indices = tf.stack([tf.range(len(actions)), actions], axis=1)
            logits_pi = tf.gather_nd(logits_prob, action_indices)

            policy_loss = -tf.reduce_mean(logits_pi * returns)

            probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_sum(probs * logits_prob, axis=1)
            entropy_loss = -self.entropy_beta * tf.reduce_mean(entropy)
            loss = policy_loss + entropy_loss

        grads = tape.gradient(loss, self.model.trainable_variables)

        clipped_grads, grad_norm = tf.clip_by_global_norm(grads, GRADIENT_CLIP_NORM)
        self.model.optimizer.apply_gradients(zip(clipped_grads, self.model.trainable_variables))

        return {
            "loss": float(loss.numpy()),
            "entropy_loss": float(entropy_loss.numpy()),
            "policy_loss": float(policy_loss.numpy()),
            "grad_norm": float(grad_norm.numpy()),
            "mean_entropy": float(tf.reduce_mean(entropy).numpy()),
        }

    @staticmethod
    def discount_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
        """
        Compute discounted Monte Carlo returns via backward pass through episode.

        Calculates the discounted cumulative reward G_t for each timestep t in an episode
        using the formula: G_t = Σ_{k=0}^{T-t} γ^k * r_{t+k}

        This is the "return" or "cumulative discounted reward" from timestep t onward,
        which represents the total expected future reward when starting from that timestep.
        The backward pass efficiently computes all returns in O(T) time.

        Args:
            rewards (np.ndarray): Array of immediate rewards for one episode.
                Shape: (episode_length,)
                Values: r_0, r_1, ..., r_{T-1} where T is episode length
                Example: [0.1, 0.0, 0.0, 1.0] for a 4-step episode
            gamma (float): Discount factor for future rewards, range [0, 1].
                - gamma = 0: Only immediate reward matters (G_t = r_t)
                - gamma = 1: All future rewards count equally (undiscounted sum)
                - gamma ∈ (0, 1): Exponentially decaying weight on future rewards
                Typical values: 0.95-0.99

        Returns:
            np.ndarray: Array of discounted returns for each timestep.
                Shape: (episode_length,)
                dtype: float32
                Values: G_0, G_1, ..., G_{T-1}
        """
        returns = np.zeros_like(rewards, dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + gamma * running
            returns[t] = running
        return returns

    def run(self, epochs: int) -> None:
        """
        Execute the main REINFORCE training loop for the specified number of epochs.

        Runs the complete training process, managing episode collection, batch accumulation,
        policy updates, logging, and model checkpointing. Each epoch represents one full
        episode in the environment. Episodes are accumulated into batches before updating
        the policy network.

        Training Flow Per Epoch:
            1. Run simulation() to collect one episode trajectory
            2. Compute discounted returns G_t for the episode
            3. Append episode data to current batch buffers
            4. If batch is complete (epoch % episodes_per_batch == 0):
               a. Concatenate all episodes in batch
               b. Call fit_model() to update policy
               c. Clear batch buffers
            5. Log metrics to TensorBoard (reward, losses, entropy, gradients)
            6. Reset environment for next episode
            7. Save model checkpoint every MODEL_SAVE_FREQUENCY epochs

        Batch Management:
            Episodes are accumulated until episodes_per_batch is reached, then:
            - All observations are concatenated into shape (total_timesteps, obs_dim)
            - All actions are concatenated into shape (total_timesteps,)
            - All returns are concatenated into shape (total_timesteps,)
            - Policy update is performed on the entire batch
            - Batch buffers are cleared and accumulation restarts

        Args:
            epochs (int): Total number of training episodes to run.
                Each epoch = 1 complete episode in the environment.
                Typical values: 1000-5000 depending on task complexity.
                Note: Policy updates happen every episodes_per_batch epochs,
                not every epoch.

        TensorBoard Metrics Logged (per epoch):
            - "Reinforce/Reward": Total undiscounted reward for the episode
            - "Reinforce/Loss": Total loss (policy + entropy) from last update
            - "Reinforce/Policy_Loss": Policy gradient loss from last update
            - "Reinforce/Entropy_Loss": Entropy regularization term from last update
            - "Reinforce/Gradient_Norm": L2 norm of gradients before clipping
            - "Reinforce/Mean_Entropy": Average entropy across batch (exploration metric)

        Model Checkpointing:
            - Models are saved every MODEL_SAVE_FREQUENCY epochs (default: 500)
            - Saved to: MODEL_PATH/{model_name}.keras
            - Includes full model architecture, weights, and optimizer state
        """

        batch_observations = []
        batch_actions = []
        batch_returns = []

        metrics = {
            "loss": 0.0,
            "entropy_loss": 0.0,
            "policy_loss": 0.0,
            "grad_norm": 0.0,
            "mean_entropy": 0.0,
        }

        for epoch in range(epochs):

            observations, actions, rewards = self.simulation()
            returns = self.discount_returns(rewards, self.gamma)

            batch_observations.append(observations)
            batch_actions.append(actions)
            batch_returns.append(returns)

            if (epoch + 1) % self.episodes_per_batch == 0:
                batch_observations = np.concatenate(batch_observations, axis=0)
                batch_actions = np.concatenate(batch_actions, axis=0)
                batch_returns = np.concatenate(batch_returns, axis=0)

                metrics = self.fit_model(batch_observations, batch_actions, batch_returns)

                batch_observations = []
                batch_actions = []
                batch_returns = []

            reward = sum(rewards)

            with self.tb_writer.as_default():
                tf.summary.scalar("Reinforce/Reward", reward, epoch)
                tf.summary.scalar("Reinforce/Loss", metrics["loss"], epoch)
                tf.summary.scalar("Reinforce/Policy_Loss", metrics["policy_loss"], epoch)
                tf.summary.scalar("Reinforce/Entropy_Loss", metrics["entropy_loss"], epoch)
                tf.summary.scalar("Reinforce/Gradient_Norm", metrics["grad_norm"], epoch)
                tf.summary.scalar("Reinforce/Mean_Entropy", metrics["mean_entropy"], epoch)

            self.environment.reset()

            logger().info(f"Epoch {epoch + 1}/{epochs} completed with reward {reward}")

            if epoch % MODEL_SAVE_FREQUENCY == 0 and epoch != 0:
                self.save_model()

        self.close_tb()

    def save_model(self) -> None:
        """
        Save the trained policy network to disk.

        Persists the complete Keras model including architecture, weights, optimizer state,
        and compilation configuration to a .keras file. This allows for later loading and
        continuation of training or deployment for inference in production.

        The model is saved in Keras 3 format (.keras) which is a single HDF5 file containing:
        - Model architecture (layers, connections, input/output shapes)
        - Trained weights (all parameters for all layers)
        - Optimizer state (e.g., Adam momentum terms, iteration count)
        - Loss and metrics configuration
        - Training configuration (learning rate, etc.)

        File Location:
            MODEL_PATH/{model_name}.keras
        """
        path = os.path.join(MODEL_PATH, self.model_name + ".keras")
        self.model.save(path)
        logger().info(f"Model saved successfully at {path}")

    @abstractmethod
    def simulation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run one complete episode in the environment (abstract method).

        This method must be implemented by subclasses to define the specific episode logic
        for the training environment. It handles the interaction loop between the agent and
        environment, including observation collection, action selection via the policy network,
        and reward accumulation.

        The simulation should:
        1. Initialize episode (reset if needed)
        2. Loop until episode termination:
           a. Get current observation from environment
           b. Select action using self.policy(observation)
           c. Execute action in environment
           d. Collect reward
           e. Store (observation, action, reward) tuple
           f. Check for episode termination
        3. Return collected trajectories

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Episode trajectory data:
                - observations (np.ndarray): State observations, shape (episode_length, obs_dim)
                - actions (np.ndarray): Actions taken, shape (episode_length,), dtype int
                - rewards (np.ndarray): Immediate rewards, shape (episode_length,), dtype float

        Raises:
            NotImplementedError: If subclass does not implement this method.
        """
        raise NotImplementedError("Method simulation() not implemented in TrainerReinforce.")
