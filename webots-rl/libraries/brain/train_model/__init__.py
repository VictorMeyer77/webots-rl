import os
from abc import ABC, abstractmethod

from brain.utils.tcp_socket import TcpSocket

MODEL_PATH = "/Users/victormeyer/Dev/Self/webots-rl/output/model"


class TrainModel(ABC):
    """
    Abstract base class for all models.

    Attributes:
        model_dir (str): Directory where model files are stored.
        name (str): Name of the model.
    """

    model_dir: str
    name: str
    tcp_socket: TcpSocket | None

    def __init__(self):
        """
        Initialize the model with a name and ensure the model directory exists.
        """
        self.name = self.get_name()
        self.model_dir = MODEL_PATH
        os.makedirs(self.model_dir, exist_ok=True)
        self.tcp_socket = TcpSocket(is_client=False)

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the name of the model.

        Returns:
            str: The model's name.
        """
        raise NotImplementedError("Method get_name() not implemented.")

    @abstractmethod
    def save(self):
        """
        Save the model to disk.
        """
        raise NotImplementedError("Method save() not implemented.")
