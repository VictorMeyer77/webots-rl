"""
PPO (Proximal Policy Optimization) multi-environment training module.

This module implements a PPO algorithm for reinforcement learning with support
for multiple parallel environments. It uses GAE (Generalized Advantage Estimation)
to compute advantages and communicates with environments via TCP sockets.

PPO is an on-policy algorithm that uses a clipped surrogate objective to prevent
large policy updates. It collects batches of experience from parallel environments,
computes advantages using GAE, and performs multiple epochs of mini-batch updates
on the collected data.

Classes:
    TrainerPPO: PPO trainer inheriting from MultiTrainer to manage multiple
                environments simultaneously.

Constants:
    MODEL_SAVE_FREQUENCY_MINUTES: Model save frequency in minutes (10).
"""

import time

import brain.utils.tcp_socket as tcp
import numpy as np
import tensorflow as tf
from brain.multi_trainer import MultiTrainer
from brain.utils.logger import logger

MODEL_SAVE_FREQUENCY_MINUTES = 10


class TrainerPPO(MultiTrainer):
    """
    PPO trainer for multi-environment reinforcement learning.

    This class implements the Proximal Policy Optimization (PPO) algorithm with
    Generalized Advantage Estimation (GAE) to train a model across multiple
    parallel environments. PPO is an on-policy algorithm that uses a clipped
    surrogate objective to limit policy updates and improve training stability.

    The trainer collects experience from multiple environments simultaneously,
    computes advantages using GAE, and performs multiple epochs of updates on
    mini-batches of the collected data. This approach improves sample efficiency
    while maintaining training stability through policy clipping.

    Attributes:
        num_actions (int): Number of possible actions in the environment.
        fit_step_frequency (int): Number of steps to collect before performing
            a model update. This determines the batch size for PPO updates.
        gamma (float): Discount factor for future rewards. Values closer to 1
            make the agent more far-sighted, while values closer to 0 make it
            more myopic. Typical values: 0.95-0.99.
        entropy_coefficient (float): Coefficient for the entropy bonus term in
            the loss function. Higher values encourage more exploration by
            preventing premature convergence to deterministic policies.
            Typical values: 0.01-0.1.
        value_loss_coefficient (float): Coefficient for the value function loss
            in the total loss. Balances the importance of value estimation
            relative to policy improvement. Typical values: 0.5-1.0.
        grad_norm_clip (float): Maximum norm for gradient clipping. Prevents
            excessively large gradient updates that can destabilize training.
            Typical values: 0.5-10.0.
        lambda_ (float): Lambda parameter for GAE (Generalized Advantage
            Estimation). Controls the bias-variance tradeoff in advantage
            estimation. Values closer to 0 give lower variance but higher bias,
            while values closer to 1 give lower bias but higher variance.
            Typical values: 0.95-0.99.
        ppo_epochs (int): Number of training epochs to perform on each batch
            of collected data. Higher values improve sample efficiency but may
            lead to overfitting. Typical values: 3-10.
        mini_batch_size (int): Size of mini-batches for SGD updates. Smaller
            batches can lead to noisier but potentially more effective updates.
            Should divide evenly into (fit_step_frequency * num_env).
        clip_ratio (float): Clipping parameter for the PPO objective. Limits
            the ratio between new and old policies to prevent large updates.
            Higher values allow larger policy changes. Typical values: 0.1-0.3.
        step_count (int): Counter for the number of training steps (fit_model
            calls) performed. Used for logging and tracking training progress.
        last_model_save_time (float): Timestamp of the last model save operation.
            Used to implement periodic model checkpointing.
    """

    num_actions: int
    fit_step_frequency: int
    gamma: float
    entropy_coefficient: float
    value_loss_coefficient: float
    grad_norm_clip: float
    lambda_: float
    ppo_epochs: int
    mini_batch_size: int
    clip_ratio: float
    step_count: int = 0
    last_model_save_time: float

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
        lambda_: float,
        ppo_epochs: int,
        mini_batch_size: int,
        clip_ratio: float,
    ):
        """
        Initializes the PPO trainer with the specified hyperparameters.

        Sets up the PPO-specific parameters including the clipping ratio, number of
        epochs, mini-batch size, and GAE lambda. Also initializes the parent
        MultiTrainer class with the model, optimizer, and memory configuration.

        Args:
            model_name (str): Name identifier for the model, used for saving and
                logging purposes.
            model (tf.keras.Model): Neural network model that outputs both policy
                logits and state values. Must have two output heads.
            optimizer (tf.keras.optimizers.Optimizer): TensorFlow optimizer for
                gradient-based updates (e.g., Adam, RMSprop).
            nb_env (int): Number of parallel environments to run simultaneously.
                More environments provide more diverse experience but require more
                computational resources.
            num_actions (int): Number of discrete actions available in the
                environment's action space.
            fit_step_frequency (int): Number of steps to collect from each
                environment before performing a PPO update. The total batch size
                will be fit_step_frequency * nb_env.
            gamma (float): Discount factor for future rewards, in range [0, 1].
            entropy_coefficient (float): Weight for the entropy bonus term that
                encourages exploration.
            value_loss_coefficient (float): Weight for the value function loss in
                the total loss function.
            grad_norm_clip (float): Maximum norm for gradient clipping to prevent
                gradient explosion.
            lambda_ (float): GAE lambda parameter for advantage estimation, in
                range [0, 1].
            ppo_epochs (int): Number of epochs to train on each batch of data.
                More epochs improve sample efficiency but may cause overfitting.
            mini_batch_size (int): Size of mini-batches for stochastic gradient
                descent updates.
            clip_ratio (float): PPO clipping parameter epsilon, typically 0.1-0.3.
                Controls how much the policy can change in one update.
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
        self.lambda_ = lambda_
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.clip_ratio = clip_ratio
        self.step_count = 0
        self.last_model_save_time = time.time()

    def policy(self, observation: np.ndarray) -> tuple[int, float]:
        """
        Computes an action and its estimated value for a given observation.

        Uses the current policy network to sample an action from the policy
        distribution and estimate the state value. The action is sampled
        stochastically from the categorical distribution defined by the policy
        logits, which encourages exploration during training.

        Args:
            observation (np.ndarray): State observation from the environment,
                typically a preprocessed image or feature vector.

        Returns:
            tuple[int, float]: A tuple containing:
                - int: The selected action index sampled from the policy distribution.
                - float: The estimated state value from the value function head.
        """
        observation = np.expand_dims(observation, axis=0)
        logits, value = self.model(observation)
        action = tf.random.categorical(logits, 1)
        action = tf.squeeze(action, axis=1)
        return int(action.numpy()), float(value.numpy())

    def compute_gae(
        self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, next_value: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes GAE (Generalized Advantage Estimation) advantages and returns.

        GAE is a method for estimating advantages that reduces variance while
        maintaining reasonable bias. It uses an exponentially-weighted average of
        temporal difference (TD) errors at different time scales, controlled by
        the lambda parameter.

        The GAE formula is:
        A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t) is the TD error.

        Higher lambda values (closer to 1) reduce bias but increase variance, while
        lower values (closer to 0) reduce variance but increase bias. The returns
        are computed as advantages + values for training the value function.

        Args:
            rewards (np.ndarray): Rewards received at each time step for a single
                environment, shape (steps,).
            values (np.ndarray): Estimated state values at each time step for a
                single environment, shape (steps,).
            dones (np.ndarray): Episode termination indicators for a single
                environment, shape (steps,). Used to reset advantage computation
                at episode boundaries.
            next_value (float): Estimated value of the state following the last
                step in the trajectory. Used for bootstrapping the last advantage.
                Should be 0 if the last state is terminal.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - np.ndarray: GAE advantages for each time step, shape (steps,).
                  These are used to weight policy updates.
                - np.ndarray: Returns (advantages + values) for each time step,
                  shape (steps,). These are used as targets for value function
                  training.
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def gather_batch(self):
        """
        Gathers and organizes experience data from all environments from memory.

        Collects data stored in the memory buffer by each environment and organizes
        it into synchronized arrays where the first dimension is time steps and the
        second dimension is environments. This ensures that data from different
        environments at the same time step is aligned properly for batch processing.

        The memory is cleared (via popleft) as data is gathered, preparing it for
        the next batch of experience collection.

        Returns:
            tuple: A tuple containing five numpy arrays:
                - observations (np.ndarray): State observations with shape
                  (steps, num_env, height, width, frames). Contains the visual or
                  feature inputs to the policy.
                - actions (np.ndarray): Actions taken by the policy with shape
                  (steps, num_env). Integer action indices.
                - rewards (np.ndarray): Rewards received from the environment with
                  shape (steps, num_env). Float values.
                - dones (np.ndarray): Episode termination flags with shape
                  (steps, num_env). Float values (0 or 1).
                - values (np.ndarray): Value estimates from the value function with
                  shape (steps, num_env). Used for GAE computation.
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
        Trains the model using the PPO algorithm with collected experience data.

        This method implements the core PPO training procedure:

        1. Data Collection: Gathers experience data from all environments.

        2. Advantage Estimation: Computes advantages using GAE for each environment
           separately, properly handling episode boundaries and bootstrapping from
           the next state values.

        3. Advantage Normalization: Normalizes advantages across the entire batch
           to reduce variance and improve training stability.

        4. Old Policy Computation: Computes log probabilities under the current
           (old) policy for all actions in the batch. These are used to compute
           the importance sampling ratio in the PPO objective.

        5. Multi-Epoch Training: Performs multiple epochs of training on the
           collected data:
           - Shuffles the data to break temporal correlations
           - Processes data in mini-batches for computational efficiency
           - Computes the PPO clipped surrogate loss to limit policy updates
           - Trains the value function to predict returns accurately
           - Adds an entropy bonus to encourage exploration
           - Clips gradients to prevent instability

        6. Logging: Records training metrics to TensorBoard for monitoring:
           - Total loss and its components (policy, value, entropy)
           - Mean reward and advantage values
           - Clip fraction (proportion of updates that hit the clipping limit)

        The PPO clipped objective prevents the policy from changing too much in
        a single update by clipping the importance sampling ratio. This improves
        training stability compared to standard policy gradient methods.

        The method updates the global step counter after training for proper
        metric tracking across multiple training iterations.
        """
        observations, actions, rewards, dones, values = self.gather_batch()

        # Get next state values for proper bootstrapping
        _, last_values = self.model(tf.convert_to_tensor(observations[-1], dtype=tf.float32))
        last_values = tf.squeeze(last_values).numpy()

        # Bootstrap properly: mask terminal states
        next_values = last_values * (1 - dones[-1])

        # Use GAE instead of simple advantage estimation
        steps, num_env = rewards.shape
        advantages_gae = []
        returns_gae = []

        # Compute GAE for each environment separately
        for env_idx in range(num_env):
            adv, ret = self.compute_gae(
                rewards[:, env_idx], values[:, env_idx], dones[:, env_idx], next_values[env_idx]
            )
            advantages_gae.append(adv)
            returns_gae.append(ret)

        advantages_gae = np.array(advantages_gae).T  # (steps, num_env)
        returns_gae = np.array(returns_gae).T  # (steps, num_env)

        # Reshape for training
        _, _, height, width, frames = observations.shape
        observation_batch = observations.reshape(steps * num_env, height, width, frames)
        actions_batch = actions.reshape(steps * num_env)
        returns_batch = tf.convert_to_tensor(returns_gae.reshape(steps * num_env), dtype=tf.float32)
        advantages = advantages_gae.reshape(steps * num_env)

        # Normalize advantages
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

        # Compute old log probabilities (for PPO ratio)
        old_logits, _ = self.model(observation_batch)
        action_masks = tf.one_hot(actions_batch, self.num_actions)
        old_log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(old_logits), axis=1)
        old_log_probs = tf.stop_gradient(old_log_probs)  # Don't backprop through old policy

        batch_size = observation_batch.shape[0]
        indices = np.arange(batch_size)

        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mini_batch_indices = indices[start:end]

                obs_mini = tf.gather(observation_batch, mini_batch_indices)
                actions_mini = tf.gather(actions_batch, mini_batch_indices)
                returns_mini = tf.gather(returns_batch, mini_batch_indices)
                advantages_mini = tf.gather(advantages, mini_batch_indices)
                old_log_probs_mini = tf.gather(old_log_probs, mini_batch_indices)

                with tf.GradientTape() as tape:
                    logits, predicted_values = self.model(obs_mini)
                    predicted_values = tf.squeeze(predicted_values, axis=-1)

                    # Compute new log probabilities
                    action_masks_mini = tf.one_hot(actions_mini, self.num_actions)
                    new_log_probs = tf.reduce_sum(action_masks_mini * tf.nn.log_softmax(logits), axis=1)

                    # PPO clipped surrogate loss
                    ratio = tf.exp(new_log_probs - old_log_probs_mini)
                    clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                    policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages_mini, clipped_ratio * advantages_mini))

                    # Value loss
                    value_loss = tf.reduce_mean(tf.square(returns_mini - predicted_values))

                    # Entropy bonus
                    pi = tf.nn.softmax(logits)
                    entropy = -tf.reduce_mean(tf.reduce_sum(pi * tf.math.log(pi + 1e-8), axis=1))

                    loss = policy_loss + value_loss * self.value_loss_coefficient - entropy * self.entropy_coefficient

                grads = tape.gradient(loss, self.model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, self.grad_norm_clip)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Log metrics (once per fit_model call)
        with self.tb_writer.as_default():
            tf.summary.scalar("PPO/Loss", loss, step=self.step_count)
            tf.summary.scalar("PPO/Policy_Loss", policy_loss, step=self.step_count)
            tf.summary.scalar("PPO/Value_Loss", value_loss, step=self.step_count)
            tf.summary.scalar("PPO/Entropy", entropy, step=self.step_count)
            tf.summary.scalar("PPO/Reward_Mean", tf.reduce_mean(rewards), step=self.step_count)
            tf.summary.scalar("PPO/Advantage_Mean", tf.reduce_mean(advantages), step=self.step_count)
            tf.summary.scalar(
                "PPO/Clip_Fraction",
                tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > self.clip_ratio, tf.float32)),
                step=self.step_count,
            )

        self.step_count += 1

    def run(self) -> None:
        """
        Main communication loop that handles environment interactions and triggers training.

        This method implements the main training loop that coordinates between multiple
        parallel environments and the PPO trainer:

        1. Connection Setup: Accepts TCP socket connections from all configured
           environments, waiting until all are connected.

        2. Message Processing: Continuously polls all environment connections for
           incoming messages and handles three types of requests:

           - "policy": Environment requests an action for a given observation.
             The trainer computes an action using the current policy and sends it
             back along with the state value estimate.

           - "step": Environment reports the results of taking an action (new
             observation, reward, done flag, value). This data is stored in memory
             for later training.

           - "terminated": Environment signals that it has completed its episodes
             and is shutting down. The connection is closed and removed from the
             active socket list.

        3. Training Trigger: When enough experience has been collected
           (memory_count reaches fit_step_frequency), triggers a PPO training
           update by calling fit_model().

        4. Periodic Saving: Saves the model periodically (every
           MODEL_SAVE_FREQUENCY_MINUTES minutes) to create checkpoints during
           long training runs.

        5. Termination: When all environments have terminated, saves the final
           model and exits the training loop.

        This non-blocking architecture allows efficient parallel data collection
        from multiple environments while periodically performing batch updates
        on the accumulated experience.
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
