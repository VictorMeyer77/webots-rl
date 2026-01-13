import json
import socket
import time
from abc import ABC, abstractmethod

import brain.utils.tcp_socket as tcp
import numpy as np
from brain.environment import Environment
from brain.utils.logger import logger

TCP_HOST = "127.0.0.1"
SOCKET_TIMEOUT = 0.001


class MultiTrainerAgent(ABC):

    environment: Environment
    connection: socket.socket | None = None
    tcp_port: int

    def __init__(self, environment: Environment, tcp_port: int):

        self.environment = environment
        self.tcp_port = tcp_port
        self._init_client()

    def _init_client(self):

        for _ in range(10):
            try:
                self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.connection.connect((TCP_HOST, self.tcp_port))
                self.connection.settimeout(SOCKET_TIMEOUT)
                logger().info("TCP client: connected with server")
                return
            except ConnectionRefusedError:
                logger().info("TCP client: waiting for TCP server...")
                time.sleep(1)
        raise ConnectionError(f"Could not connect to TCP server ({TCP_HOST}:{self.tcp_port}) after multiple attempts")

    @abstractmethod
    def simulation(self) -> float:
        raise NotImplementedError("Not imple")

    def run(self, epochs: int) -> None:

        for epoch in range(epochs):

            reward = self.simulation()
            self.environment.reset()
            logger().info(f"Epoch {epoch + 1}/{epochs} completed with reward {reward}")

        self.tcp_send_terminated()

    def tcp_send_observation(self, observation: np.ndarray) -> None:
        message = json.dumps({"message_type": "policy", "observation": observation.tolist()})
        tcp.send(self.connection, message)

    def tcp_send_step(self, observation: np.ndarray, action: int, reward: float, done: bool, value: float) -> None:
        message = json.dumps(
            {
                "message_type": "step",
                "observation": observation.tolist(),
                "action": action,
                "reward": reward,
                "done": done,
                "value": value,
            }
        )
        tcp.send(self.connection, message)

    def tcp_send_terminated(self) -> None:
        message = json.dumps(
            {
                "message_type": "terminated",
            }
        )
        tcp.send(self.connection, message)
