"""
Base trainer class for reinforcement learning models in Webots environments.

This module defines the abstract `Trainer` class, which manages model saving,
TensorBoard logging, and provides an interface for training and saving models.
Subclasses should implement the `run` and `save_model` methods.
"""

import os
import random
import string
from abc import ABC, abstractmethod

from brain.environment import Environment
from brain.utils.logger import logger
from torch.utils.tensorboard import SummaryWriter

MODEL_PATH = "/Users/victormeyer/Dev/Self/webots-rl/output/model"
TENSORBOARD_PATH = "/Users/victormeyer/Dev/Self/webots-rl/output/train"


class Trainer(ABC):
    """
    Abstract base class for RL trainers.

    Handles model naming, saving paths, and TensorBoard logging setup.
    Subclasses must implement `run` and `save_model`.

    Attributes:
        environment: The simulation environment.
        model_name: Unique name for the model instance.
        model_path: Filesystem path for saving the model.
        tb_writer: TensorBoard SummaryWriter for logging.
    """

    environment: Environment
    model_name: str
    model_path: str
    tb_writer: SummaryWriter

    def __init__(self, model_name: str, environment: Environment):
        """
        Initialize the trainer, set up model paths and TensorBoard logging.

        Parameters:
            model_name: Base name for the model.
            environment: The simulation environment instance.
        """
        self.environment = environment
        self.model_name = f"{model_name}_{''.join(random.choices(string.ascii_letters + string.digits, k=4))}"
        self.model_path = os.path.join(MODEL_PATH, self.model_name + ".npy")

        os.makedirs(MODEL_PATH, exist_ok=True)
        tensorboard_dir = os.path.join(TENSORBOARD_PATH, self.model_name)
        self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)
        logger().info(f"TensorBoard logging to {tensorboard_dir}")

    def close_tb(self):
        """
        Flush and close the TensorBoard SummaryWriter.
        """
        self.tb_writer.flush()
        self.tb_writer.close()

    @abstractmethod
    def save_model(self):
        """
        Save the model to disk.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Method save() not implemented.")

    @abstractmethod
    def run(self, epochs: int):
        """
        Run the training process for a given number of epochs.

        Parameters:
            epochs: Number of training iterations.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Method run() not implemented.")
