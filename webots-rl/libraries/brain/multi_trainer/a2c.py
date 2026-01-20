"""
A2C (Advantage Actor-Critic) multi-environment trainer module.

This module implements the Advantage Actor-Critic (A2C) algorithm for parallel
reinforcement learning across multiple simulator instances communicating via TCP.

Architecture:
  * Actor-Critic Network: A single neural network outputs both policy logits
    (actor) and state value estimates (critic).
  * Parallel Environments: Multiple simulation instances run concurrently,
    each sending observations and receiving actions through TCP sockets.
  * Synchronous Updates: After collecting a batch of transitions from all
    environments, the model is updated using computed advantages.

Key Features:
  * Generalized Advantage Estimation (GAE) via bootstrapped returns.
  * Gradient clipping to prevent instability.
  * TensorBoard logging of losses, rewards, and advantages.
  * Periodic model checkpointing (every 10 minutes).
  * Graceful shutdown when all environments terminate.

Notes:
  * The model must return (logits, value) where logits shape is [batch, num_actions].
  * Observations are expected to be normalized externally before being sent.
  * Advantages are standardized internally to stabilize training.
  * All environments share the same policy parameters (synchronous updates).
"""

import time

import brain.utils.tcp_socket as tcp
import numpy as np
import tensorflow as tf
from brain.multi_trainer import MultiTrainer
from brain.utils.logger import logger

MODEL_SAVE_FREQUENCY_MINUTES = 10


class TrainerA2C(MultiTrainer):
    """
    A2C trainer for multi-environment reinforcement learning.

    This class implements the Advantage Actor-Critic (A2C) algorithm with support
    for parallel training across multiple environments. It manages experience
    collection, discounted return computation, and model updates using the
    actor-critic architecture.

    The trainer operates in a server mode, accepting TCP connections from multiple
    simulation environments and coordinating their interactions with a shared
    neural network policy.

    Attributes:
        num_actions (int): Number of discrete actions in the action space.
        fit_step_frequency (int): Number of steps to collect before performing
            a model update. Determines batch size (fit_step_frequency × nb_env).
        gamma (float): Discount factor for computing returns (0 < gamma ≤ 1).
            Typical values: 0.99 for long-horizon tasks, 0.95 for shorter ones.
        entropy_coefficient (float): Weight for entropy regularization term.
            Higher values encourage more exploration. Typical range: [0.001, 0.1].
        value_loss_coefficient (float): Weight for value function loss.
            Balances critic learning against policy learning. Typical: 0.5.
        grad_norm_clip (float): Maximum norm for gradient clipping. Prevents
            exploding gradients. Typical range: [0.5, 5.0].
        step_count (int): Counter tracking the number of model updates performed.
            Used for TensorBoard step indexing.
    """

    num_actions: int
    fit_step_frequency: int
    gamma: float
    entropy_coefficient: float
    value_loss_coefficient: float
    grad_norm_clip: float
    step_count: int = 0

    def __init__(
        self,
        model_name: str,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        nb_env: int,
        num_actions: int,
        fit_step_frequency: int,
        gamma: float,
        entropy_coefficient: float,
        value_loss_coefficient: float,
        grad_norm_clip: float,
    ):
        """
        Initialize the A2C trainer with hyperparameters and network architecture.

        Sets up the training infrastructure including memory buffers for each
        environment, TensorBoard logging, and model checkpointing.

        Args:
            model_name (str): Identifier for the model used in saving checkpoints
                and TensorBoard logs. Should be descriptive (e.g., "cartpole_a2c").
            model (tf.keras.Model): Actor-Critic neural network. Must return a
                tuple (logits, value) where:
                  - logits: Tensor of shape [batch, num_actions] (unnormalized action probs).
                  - value: Tensor of shape [batch, 1] (state value estimate).
            optimizer (tf.keras.optimizers.Optimizer): TensorFlow optimizer for
                gradient descent. Common choice: Adam with learning_rate=7e-4.
            nb_env (int): Number of parallel environments. More environments
                provide more diverse experiences but require more memory.
            num_actions (int): Size of the discrete action space. Must match
                the output dimension of the actor head.
            fit_step_frequency (int): Number of transitions to collect per
                environment before updating. Total batch size will be
                (fit_step_frequency × nb_env). Typical values: [5, 20].
            gamma (float): Discount factor γ ∈ (0, 1] for computing returns.
                Higher values prioritize long-term rewards.
            entropy_coefficient (float): Coefficient β for entropy regularization.
                Entropy term: -β × mean(Σ π log π). Encourages exploration.
            value_loss_coefficient (float): Coefficient c for value loss.
                Total loss includes: c × MSE(V(s), R). Balances critic training.
            grad_norm_clip (float): Maximum L2 norm for gradients. Clips all
                gradients jointly to prevent instability from large updates.
        """
        super().__init__(
            model_name=model_name, model=model, optimizer=optimizer, nb_env=nb_env, memory_size=fit_step_frequency * 2
        )
        self.num_actions = num_actions
        self.fit_step_frequency = fit_step_frequency
        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient
        self.value_loss_coefficient = value_loss_coefficient
        self.grad_norm_clip = grad_norm_clip
        self.step_count = 0
        self.last_model_save_time = time.time()

    def policy(self, observation: np.ndarray) -> tuple[int, float]:
        """
        Select an action based on the current observation using the stochastic policy.

        Uses the actor-critic network to compute action probabilities (via softmax
        over logits) and samples an action from this distribution. Also returns
        the value estimate for the state.

        This implements the stochastic policy π(a|s) = softmax(logits(s)), where
        actions are sampled rather than greedily selected. This ensures exploration
        during training.

        Args:
            observation (np.ndarray): Current state observation from the environment.
                Shape depends on the task (e.g., [height, width, channels] for images
                or [features] for vector observations). Must match the model's
                expected input shape.

        Returns:
            tuple[int, float]: A tuple containing:
                - action (int): Discrete action sampled from π(·|s). Index in [0, num_actions).
                - value (float): Critic's estimate V(s) of the state value. Used for
                  computing advantages and as the bootstrap value for return calculation.
        """
        observation = np.expand_dims(observation, axis=0)
        logits, value = self.model(observation)
        action = tf.random.categorical(logits, 1)
        action = tf.squeeze(action, axis=1)
        return int(action.numpy()), float(value.numpy())

    def compute_returns(self, rewards: np.ndarray, dones: np.ndarray, last_values: np.ndarray) -> np.ndarray:
        """
        Compute discounted returns using the Bellman equation with bootstrapping.

        Applies the recursive formula backwards from the final step:
            R_t = r_t + γ × R_{t+1} × (1 - done_t)

        where:
          - R_t is the return at time t
          - r_t is the immediate reward
          - γ is the discount factor
          - done_t is 1 if the episode terminated at step t, else 0

        The last_values parameter provides bootstrapping for non-terminal final
        states, allowing the returns to incorporate the critic's value estimate
        when episodes don't finish within the batch window.

        Args:
            rewards (np.ndarray): Immediate rewards received at each step.
                Shape: [steps, num_env]. Each element r_{t,i} is the reward
                for environment i at timestep t.
            dones (np.ndarray): Episode termination flags. Shape: [steps, num_env].
                Element is 1.0 if episode ended, 0.0 otherwise. When done=1,
                future returns are not considered (R_{t+1} term is zeroed).
            last_values (np.ndarray): Bootstrapped value estimates for the final
                state. Shape: [num_env]. Used when episodes don't terminate
                within the batch: R_T = V(s_T) if not done, else R_T = 0.

        Returns:
            np.ndarray: Discounted returns for each timestep and environment.
                Shape: [steps, num_env]. Each R_{t,i} represents the total
                discounted reward from step t onward for environment i.
        """
        returns = []
        r = last_values
        for t in reversed(range(len(rewards))):
            r = rewards[t] + self.gamma * r * (1 - dones[t])
            returns.insert(0, r)
        return np.array(returns)

    def gather_batch(self):
        """
        Extract and organize transitions from all environment memories into batches.

        Collects experiences stored during the policy execution phase and structures
        them into numpy arrays suitable for batched neural network training. The
        method ensures temporal alignment across all environments by gathering
        transitions step-by-step.

        The memory is organized as a dictionary mapping ports to deques of transitions.
        Each transition is a tuple: (observation, action, reward, done, value).

        Returns:
            tuple: A 5-tuple of numpy arrays containing:
                - observations (np.ndarray): States visited. Shape: [steps, num_env,
                  height, width, frames]. For non-image observations, the spatial
                  dimensions may differ.
                - actions (np.ndarray): Actions taken. Shape: [steps, num_env].
                  Each element is an integer action index.
                - rewards (np.ndarray): Rewards received. Shape: [steps, num_env].
                  Immediate scalar rewards for each transition.
                - dones (np.ndarray): Termination flags. Shape: [steps, num_env].
                  Binary indicators (0 or 1) for episode completion.
                - values (np.ndarray): Value estimates at each state. Shape:
                  [steps, num_env]. Critic's predictions V(s_t) stored during
                  policy execution, used for advantage calculation.

        Notes:
            * The memory is cleared (deque.popleft) during gathering, making it
              a destructive operation. After calling this method, the memory
              buffers are empty and ready for the next batch collection.
            * Ports are sorted to ensure consistent ordering across calls.
            * All environments must have the same number of transitions in memory.
        """
        ports = sorted(self.memory.keys())
        steps = self.memory_count()

        observations, actions, rewards, dones, values = [], [], [], [], []

        for step in range(steps):
            step_observation, step_action, step_reward, step_done, step_value = [], [], [], [], []

            for port in ports:
                observation, action, reward, done, value = self.memory[port].popleft()
                step_observation.append(observation)
                step_action.append(action)
                step_reward.append(reward)
                step_done.append(done)
                step_value.append(value)

            observations.append(step_observation)
            actions.append(step_action)
            rewards.append(step_reward)
            dones.append(step_done)
            values.append(step_value)

        return (
            np.array(observations, dtype=np.float32),  # (steps, num_env, obs_dim)
            np.array(actions, dtype=np.int32),  # (steps, num_env)
            np.array(rewards, dtype=np.float32),  # (steps, num_env)
            np.array(dones, dtype=np.float32),  # (steps, num_env)
            np.array(values, dtype=np.float32),  # (steps, num_env)
        )

    def fit_model(self) -> None:
        """
        Perform a single model update using collected experiences via gradient descent.

        This method implements the core A2C update algorithm:
          1. Gather batch of transitions from all environments.
          2. Compute bootstrapped returns using the Bellman equation.
          3. Calculate advantages: A(s,a) = R(s,a) - V(s).
          4. Standardize advantages for numerical stability.
          5. Compute three loss components:
             - Policy loss: encourages actions with positive advantages.
             - Value loss: trains critic to predict returns accurately.
             - Entropy loss: regularizes policy to maintain exploration.
          6. Apply gradient clipping to prevent instability.
          7. Update network weights using the optimizer.
          8. Log metrics to TensorBoard for monitoring.

        Loss Definitions:
          * Policy Loss (L_π):
              L_π = -mean(log π(a_t | s_t) × A_t)
            where A_t are standardized advantages. Minimizing this loss increases
            the probability of actions that led to positive advantages.

          * Value Loss (L_V):
              L_V = mean((R_t - V(s_t))²)
            Mean squared error between computed returns and critic predictions.
            Trains the value function to accurately estimate future returns.

          * Entropy Loss (L_H):
              L_H = -mean(Σ_a π(a|s) × log π(a|s))
            Entropy of the policy distribution. Higher entropy means more uniform
            action distribution, encouraging exploration.

          * Total Loss:
              L = L_π + c_v × L_V - c_h × L_H
            where c_v = value_loss_coefficient, c_h = entropy_coefficient.

        Training Steps:
          1. Bootstrap last values for non-terminal states using critic.
          2. Compute returns via temporal difference (Bellman equation).
          3. Flatten batches for efficient GPU computation.
          4. Standardize advantages: A_norm = (A - mean(A)) / (std(A) + ε).
          5. Forward pass: obtain logits and value predictions.
          6. Compute losses using sampled actions and advantages.
          7. Backward pass: compute gradients via TensorFlow's GradientTape.
          8. Clip gradients by global norm to prevent divergence.
          9. Apply gradients to update network weights.
          10. Log all metrics to TensorBoard.

        Side Effects:
            * Clears the memory buffer (via gather_batch).
            * Increments self.step_count.
            * Writes metrics to TensorBoard.
            * Updates model.trainable_variables in-place.

        TensorBoard Metrics:
            * A2C/Loss: Total combined loss.
            * A2C/Policy_Loss: Actor loss (log probability × advantage).
            * A2C/Value_Loss: Critic loss (MSE of returns vs predictions).
            * A2C/Entropy: Policy entropy (exploration measure).
            * A2C/Reward_Mean: Average immediate reward across batch.
            * A2C/Advantage_Mean: Average standardized advantage.

        Notes:
            * Gradient clipping uses global norm clipping (all gradients scaled
              jointly) rather than per-parameter clipping.
            * Advantage standardization prevents large magnitude advantages from
              dominating the policy loss.
            * The +1e-8 constants prevent division by zero in standardization
              and log calculations.
        """
        observations, actions, rewards, dones, values = self.gather_batch()

        _, last_values = self.model(tf.convert_to_tensor(observations[-1], dtype=tf.float32))
        last_values = tf.squeeze(last_values).numpy()

        returns = self.compute_returns(rewards, dones, last_values)

        steps, num_env, height, width, frames = observations.shape
        observation_batch = observations.reshape(steps * num_env, height, width, frames)
        actions_batch = actions.reshape(steps * num_env)
        returns_batch = tf.convert_to_tensor(returns.reshape(steps * num_env), dtype=tf.float32)
        values_batch = values.reshape(steps * num_env)
        advantages = returns_batch - values_batch
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

        with tf.GradientTape() as tape:
            logits, predicted_values = self.model(observation_batch)
            predicted_values = tf.squeeze(predicted_values, axis=-1)

            action_masks = tf.one_hot(actions_batch, self.num_actions)
            log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)

            policy_loss = -tf.reduce_mean(log_probs * advantages)
            value_loss = tf.reduce_mean(tf.square(returns_batch - predicted_values))
            pi = tf.nn.softmax(logits)
            entropy = -tf.reduce_mean(tf.reduce_sum(pi * tf.math.log(pi + 1e-8), axis=1))

            loss = policy_loss + value_loss * self.value_loss_coefficient - entropy * self.entropy_coefficient

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.grad_norm_clip)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        with self.tb_writer.as_default():
            tf.summary.scalar("A2C/Loss", loss, step=self.step_count)
            tf.summary.scalar("A2C/Policy_Loss", policy_loss, step=self.step_count)
            tf.summary.scalar("A2C/Value_Loss", value_loss, step=self.step_count)
            tf.summary.scalar("A2C/Entropy", entropy, step=self.step_count)
            tf.summary.scalar("A2C/Reward_Mean", tf.reduce_mean(rewards), step=self.step_count)
            tf.summary.scalar("A2C/Advantage_Mean", tf.reduce_mean(advantages), step=self.step_count)

        self.step_count += 1

    def run(self) -> None:
        """
        Execute the main training loop coordinating multiple environments via TCP.

        This method implements a server-based training architecture where the trainer
        acts as a central coordinator for multiple simulation environments. Each
        environment connects via TCP and communicates using a JSON message protocol.

        Training Loop Architecture:
          1. Connection Phase: Accept TCP connections from nb_env environments.
          2. Message Processing Loop:
             a. Poll all connected sockets for incoming messages.
             b. Handle three message types: "policy", "step", "terminated".
             c. Accumulate experiences in memory buffers.
          3. Update Phase: Trigger fit_model() when enough transitions collected.
          4. Persistence: Save model checkpoints periodically.
          5. Termination: Exit when all environments disconnect.

        Training Triggers:
          * fit_model() is called when memory_count() == fit_step_frequency.
          * Model checkpoint saved every MODEL_SAVE_FREQUENCY_MINUTES (10 minutes).
          * Final checkpoint saved when all environments terminate.

        Concurrency Model:
          * The loop is single-threaded and polls sockets sequentially.
          * Each environment runs in its own process/thread, sending messages
            asynchronously to the trainer.
          * Non-blocking socket reads (tcp.read returns None if no data available).
          * Environments proceed independently; synchronization happens implicitly
            through the fit_step_frequency batch size requirement.

        Error Handling:
          * Disconnected sockets are gracefully removed from the active set.
          * Malformed JSON messages are logged (via tcp module).
          * The loop continues as long as at least one environment is active.

        Side Effects:
            * Opens server socket and accepts connections.
            * Continuously polls network sockets.
            * Writes model checkpoints to disk.
            * Writes TensorBoard logs.
            * Blocks until all environments terminate.

        Notes:
            * This method blocks indefinitely until training completes.
            * Environments must implement the message protocol correctly.
            * The trainer does not send unsolicited messages; it only responds.
            * Socket cleanup is handled automatically via the context manager.
        """

        self.accept_connections()

        sockets = self.sockets

        while True:

            closed_sockets = []

            for conn, port in sockets:
                obj = tcp.read(conn)

                if obj is not None:
                    message_type = obj["message_type"]

                    if message_type == "policy":
                        observation = np.array(obj["observation"])
                        action, value = self.policy(observation)
                        tcp.send(conn, {"action": action, "value": value})

                    elif message_type == "step":
                        self.memory_append(
                            port, np.array(obj["observation"]), obj["action"], obj["reward"], obj["done"], obj["value"]
                        )

                    elif message_type == "terminated":
                        closed_sockets.append(port)
                        logger().info(f"Environment on port {port} has terminated.")

            sockets = [s for s in sockets if s[1] not in closed_sockets]

            if len(sockets) == 0:
                self.save_model()
                break

            if self.memory_count() == self.fit_step_frequency:
                self.fit_model()

            if self.step_count > 0 and time.time() - self.last_model_save_time >= MODEL_SAVE_FREQUENCY_MINUTES * 60:
                self.save_model()
                self.last_model_save_time = time.time()
