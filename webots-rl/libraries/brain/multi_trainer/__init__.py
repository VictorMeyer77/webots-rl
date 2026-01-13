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

    model_name: str
    model: tf.keras.models.Model | None
    optimizer: tf.keras.optimizers.Optimizer | None
    tb_writer: tf.summary.SummaryWriter
    sockets: list[tuple[socket.socket, int]] = []
    memory_size: int
    memory: dict[int, deque]

    def __init__(
        self,
        model_name: str,
        model: tf.keras.models.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        nb_env: int,
        memory_size: int,
    ):

        self.model_name = f"{model_name}_{''.join(random.choices(string.ascii_letters + string.digits, k=4))}"
        self.model = model
        self.optimizer = optimizer

        self._init_tcp_sockets(nb_env)
        self._init_tb()

        self.memory_size = memory_size
        self.memory_clear()
        print(self.memory)

    def _init_tcp_sockets(self, nb_env: int) -> None:
        for index in range(nb_env):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((TCP_HOST, 0))
            sock.listen(1)
            port = sock.getsockname()[1]
            self.sockets.append((sock, int(port)))
            logger().info(f"TCP sock {index}: listening on {TCP_HOST}:{port}")

    def accept_connections(self) -> None:
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
        tensorboard_dir = os.path.join(TENSORBOARD_PATH, self.model_name)
        self.tb_writer = tf.summary.create_file_writer(tensorboard_dir)
        logger().info(f"TensorBoard logging to {tensorboard_dir}")

    def close_tb(self) -> None:
        """
        Flush and close the TensorBoard writer.

        Should be called after training completes to ensure all events are persisted.
        """
        self.tb_writer.flush()
        self.tb_writer.close()

    def save_model(self) -> None:
        path = os.path.join(MODEL_PATH, self.model_name + ".keras")
        self.model.save(path)
        logger().info(f"Model saved successfully at {path}")

    def memory_append(
        self, port: int, observation: np.ndarray, action: int, reward: float, done: bool, value: float
    ) -> None:
        self.memory[port].append((observation, action, reward, done, value))

    def memory_clear(self) -> None:
        self.memory = {port: deque(maxlen=self.memory_size) for _, port in self.sockets}

    def memory_count(self) -> int:
        return min([len(self.memory[port]) for port in self.memory.keys()])

    @abstractmethod
    def policy(self, observation: np.ndarray) -> int | float:
        # logits, value = self.model(observation)
        # action = tf.random.categorical(logits, 1)
        # action = tf.squeeze(action, axis=1)
        # return action.numpy()
        raise NotImplementedError("Method policy() not implemented in MultiTrainer.")

    @abstractmethod
    def fit_model(self) -> None:
        raise NotImplementedError("Method fit_model() not implemented in MultiTrainer.")

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError("Method train() not implemented in MultiTrainer.")
