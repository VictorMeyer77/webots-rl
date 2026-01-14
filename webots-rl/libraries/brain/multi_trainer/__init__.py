"""
Multi-environment trainer base module for reinforcement learning.

This module provides the abstract base class for implementing multi-environment
reinforcement learning trainers. It handles TCP socket communication with multiple
parallel environments, manages experience replay memory, TensorBoard logging,
and model persistence.

The MultiTrainer class serves as a foundation for specific RL algorithms (A2C, PPO...)
by implementing common infrastructure while leaving algorithm-specific methods
(policy, fit_model, run) to be implemented by subclasses.

Constants:
    MODEL_PATH: Directory path for saving trained models.
    TENSORBOARD_PATH: Directory path for TensorBoard logs.
    TCP_HOST: Host address for TCP socket communication (localhost).
    SOCKET_TIMEOUT: Socket timeout in seconds for non-blocking operations (0.001s).
"""
import os
import random
import socket
import string
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import tensorflow as tf
from brain.utils.logger import logger

MODEL_PATH = "/Users/victormeyer/Dev/Self/webots-rl/output/model"
TENSORBOARD_PATH = "/Users/victormeyer/Dev/Self/webots-rl/output/train"
TCP_HOST = "127.0.0.1"
SOCKET_TIMEOUT = 0.001


class MultiTrainer(ABC):
    """
    Abstract base class for multi-environment reinforcement learning trainers.

    This class provides the common infrastructure needed for training RL agents
    across multiple parallel environments. It handles:
    - TCP socket communication with environment agents
    - Experience replay memory management with per-environment buffers
    - TensorBoard integration for training metrics visualization
    - Model saving and persistence

    Subclasses must implement the abstract methods (policy, fit_model, run) to
    define the specific RL algorithm logic while inheriting the infrastructure.

    Attributes:
        model_name (str): Unique identifier for the model, composed of the base
            name plus a random 4-character suffix for versioning. Used for
            saving models and organizing TensorBoard logs.
        model (tf.keras.Model | None): The neural network model that implements
            the policy and/or value function. Architecture depends on the
            specific RL algorithm.
        optimizer (tf.keras.optimizers.Optimizer | None): TensorFlow optimizer
            (e.g., Adam, RMSprop) used for gradient-based parameter updates.
        tb_writer (tf.summary.SummaryWriter): TensorBoard writer for logging
            training metrics, losses, and performance statistics.
        sockets (list[tuple[socket.socket, int]]): List of tuples containing
            TCP socket connections and their corresponding port numbers. One
            socket per environment for bidirectional communication.
        memory_size (int): Maximum size of the experience replay buffer per
            environment. Older experiences are automatically discarded when
            the buffer is full (deque with maxlen).
        memory (dict[int, deque]): Dictionary mapping port numbers to deques
            that store experience tuples (observation, action, reward, done,
            value) for each environment. Organized by port to maintain
            per-environment memory separation.
    """
    model_name: str
    model: tf.keras.Model | None
    optimizer: tf.keras.optimizers.Optimizer | None
    tb_writer: tf.summary.SummaryWriter
    sockets: list[tuple[socket.socket, int]] = []
    memory_size: int
    memory: dict[int, deque]

    def __init__(
        self,
        model_name: str,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        nb_env: int,
        memory_size: int,
    ):
        """
        Initializes the multi-environment trainer with infrastructure setup.

        Creates TCP sockets for environment communication, initializes TensorBoard
        logging, and sets up experience replay memory buffers. The model name is
        augmented with a random suffix to enable multiple training runs without
        conflicts.

        Args:
            model_name (str): Base name for the model. A 4-character random suffix
                will be appended to create a unique identifier.
            model (tf.keras.Model): Neural network model implementing the policy
                and/or value function. The architecture should match the chosen
                RL algorithm requirements.
            optimizer (tf.keras.optimizers.Optimizer): TensorFlow optimizer for
                gradient-based parameter updates during training.
            nb_env (int): Number of parallel environments to create. Each
                environment gets its own TCP socket and memory buffer.
            memory_size (int): Maximum number of experience tuples to store per
                environment. When full, oldest experiences are discarded.
        """
        self.model_name = f"{model_name}_{''.join(random.choices(string.ascii_letters + string.digits, k=4))}"
        self.model = model
        self.optimizer = optimizer

        self._init_tcp_sockets(nb_env)
        self._init_tb()

        self.memory_size = memory_size
        self.memory_clear()

    def _init_tcp_sockets(self, nb_env: int) -> None:
        """
        Creates and binds TCP sockets for environment connections.

        Initializes one TCP socket per environment, each listening on a unique
        dynamically-assigned port. Sockets are configured with SO_REUSEADDR to
        allow quick restarts without waiting for TIME_WAIT states.

        The dynamically assigned ports (using port 0 in bind) prevent conflicts
        when running multiple trainers or when restarting after crashes.

        Args:
            nb_env (int): Number of TCP sockets to create (one per environment).
        """
        for index in range(nb_env):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((TCP_HOST, 0))
            sock.listen(1)
            port = sock.getsockname()[1]
            self.sockets.append((sock, int(port)))
            logger().info(f"TCP sock {index}: listening on {TCP_HOST}:{port}")

    def accept_connections(self) -> None:
        """
        Accepts incoming connections from all environments.

        Blocks until all configured environments have connected to their respective
        sockets. After accepting a connection, replaces the listening socket with
        the established connection socket and configures it with a short timeout
        for non-blocking polling operations.

        This method should be called after all environment processes have been
        started and are attempting to connect. It ensures all environments are
        ready before beginning training.
        """
        sockets = []
        for index, (sock, port) in enumerate(self.sockets):
            logger().info(f"TCP Waiting for connection on {TCP_HOST}:{port}")
            conn, addr = sock.accept()
            conn.settimeout(SOCKET_TIMEOUT)
            sock.close()
            logger().info(f"TCP server {index}: connected by {addr}")
            sockets.append((conn, port))
        self.sockets = sockets

    def _init_tb(self) -> None:
        """
        Initializes TensorBoard logging infrastructure.

        Creates a TensorBoard SummaryWriter that logs training metrics to a
        directory organized by model name. This enables visualization of training
        progress, loss curves, and performance metrics via TensorBoard.

        The logs directory structure is:
        TENSORBOARD_PATH/model_name/events.out.tfevents.*
        """
        tensorboard_dir = os.path.join(TENSORBOARD_PATH, self.model_name)
        self.tb_writer = tf.summary.create_file_writer(tensorboard_dir)
        logger().info(f"TensorBoard logging to {tensorboard_dir}")

    def close_tb(self) -> None:
        """
        Flushes and closes the TensorBoard writer.

        Ensures all pending TensorBoard events are written to disk before closing
        the writer. Should be called at the end of training or when the trainer
        is being shut down to prevent data loss.
        """
        self.tb_writer.flush()
        self.tb_writer.close()

    def save_model(self) -> None:
        """
        Saves the trained model to disk in Keras format.

        Persists the model's architecture, weights, optimizer state, and training
        configuration to a .keras file. The model can be loaded later for inference
        or continued training using tf.keras.models.load_model().

        The model is saved to MODEL_PATH/model_name.keras. The directory is created
        if it doesn't exist.
        """
        path = os.path.join(MODEL_PATH, self.model_name + ".keras")
        self.model.save(path)
        logger().info(f"Model saved successfully at {path}")

    def memory_append(
        self, port: int, observation: np.ndarray, action: int, reward: float, done: bool, value: float
    ) -> None:
        """
        Adds an experience tuple to an environment's memory buffer.

        Stores a complete transition (s, a, r, s', done, V(s)) in the replay
        memory for the environment identified by its port number. When the buffer
        is full (reaches memory_size), the oldest experience is automatically
        discarded (deque with maxlen).

        Args:
            port (int): Port number identifying the environment that generated
                this experience.
            observation (np.ndarray): State observation at time t, typically a
                preprocessed image or feature vector.
            action (int): Action taken by the agent at time t.
            reward (float): Reward received after taking the action.
            done (bool): Episode termination flag. True if the episode ended
                after this step.
            value (float): Value function estimate V(s) for the observation,
                used for advantage estimation in actor-critic algorithms.
        """
        self.memory[port].append((observation, action, reward, done, value))

    def memory_clear(self) -> None:
        """
        Clears all memory buffers for all environments.

        Reinitializes the memory dictionary with empty deques for each environment.
        Called after training updates to prepare for collecting fresh experience
        data. The memory_size limit is maintained for each new deque.
        """
        self.memory = {port: deque(maxlen=self.memory_size) for _, port in self.sockets}

    def memory_count(self) -> int:
        """
        Returns the minimum memory buffer size across all environments.

        Computes the smallest number of experiences stored in any environment's
        buffer. This is used to determine when enough synchronized data has been
        collected across all environments to trigger a training update.

        Training typically waits until all environments have contributed at least
        a certain number of experiences to ensure balanced data from all sources.

        Returns:
            int: Minimum number of experiences stored in any environment's buffer.
                If any environment has fewer experiences than the others, this
                returns the smallest count.
        """
        return min([len(self.memory[port]) for port in self.memory.keys()])

    @abstractmethod
    def policy(self, observation: np.ndarray) -> tuple[int, float]:
        """
        Computes an action and its estimated value from an observation using the current policy.

        This method must be implemented by subclasses to define the action selection
        strategy. For multi-environment trainers, this typically returns both:
        - An action sampled from the policy distribution (for exploration during training)
        - The estimated state value V(s) from the value function head

        The action sampling should be stochastic during training to encourage
        exploration, while evaluation may use deterministic selection (e.g., argmax
        of policy logits).

        Args:
            observation (np.ndarray): State observation from the environment,
                typically a preprocessed image or feature vector with shape
                matching the model's input requirements.

        Returns:
            tuple[int, float]: A tuple containing:
                - int: The selected action index sampled from the policy distribution
                  for discrete action spaces.
                - float: The estimated state value V(s) from the value function,
                  used for advantage estimation in actor-critic algorithms.
        """
        raise NotImplementedError("Method policy() not implemented in MultiTrainer.")

    @abstractmethod
    def fit_model(self) -> None:
        """
        Trains the model using collected experience data.

        This method must be implemented by subclasses to define the training
        procedure for the specific RL algorithm. Typically includes:
        - Retrieving experiences from memory buffers
        - Computing targets (returns, advantages, etc.)
        - Performing gradient descent updates
        - Logging training metrics to TensorBoard
        """
        raise NotImplementedError("Method fit_model() not implemented in MultiTrainer.")

    @abstractmethod
    def run(self) -> None:
        """
        Main training loop coordinating environments and training updates.

        This method must be implemented by subclasses to define the overall
        training workflow. Typically includes:
        - Accepting connections from environment agents
        - Processing incoming messages (observations, rewards, terminations)
        - Sending actions back to environments
        - Triggering fit_model() when enough data is collected
        - Handling graceful shutdown when training completes

        The communication protocol and synchronization logic depend on the
        specific requirements of the RL algorithm and environment.
        """
        raise NotImplementedError("Method train() not implemented in MultiTrainer.")
