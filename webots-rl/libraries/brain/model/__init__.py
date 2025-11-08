import os
from abc import ABC, abstractmethod

MODEL_PATH = "/Users/victormeyer/Dev/Self/webots-rl/output/model"


class Model(ABC):
    """
    Abstract base class for all models.

    Attributes:
        model_dir (str): Directory where model files are stored.
        name (str): Name of the model.
    """

    model_dir: str
    name: str

    def __init__(self):
        """
        Initialize the model with a name and ensure the model directory exists.
        """
        self.name = self.get_name()
        self.model_dir = MODEL_PATH
        os.makedirs(self.model_dir, exist_ok=True)

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

    def load(self, model_name: str):
        """
        Load the model by setting its name.

        Args:
            model_name (str): The name of the model to load.
        """
        self.name = model_name
