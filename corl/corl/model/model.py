import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class Model(ABC):
    """
    Abstract base class for all RL models.

    Defines the interface for prediction, persistence, and metadata management
    that every concrete model must implement. Subclasses are expected to
    override :meth:`predict`, :meth:`save_weights`, :meth:`load_weights` and
    :meth:`load_metadata`.
    """

    @abstractmethod
    def predict(
        self, observation: NDArray[np.float32]
    ) -> NDArray[np.int32] | NDArray[np.float32]:
        """
        Compute an action given an observation.

        Args:
            observation: Sensor/state vector from the environment.

        Returns:
            Discrete or continuous action output, depending on the action space
            of the model.
        """

    @abstractmethod
    def save_weights(self, model_dir: str) -> None:
        """
        Persist model weights to ``model_dir``.

        Args:
            model_dir: Target directory for the weight file(s).
        """

    @abstractmethod
    def load_weights(self, model_dir: str) -> None:
        """
        Restore model weights from ``model_dir``.

        Args:
            model_dir: Source directory containing the weight file(s).
        """

    def metadata(self) -> dict[str, int | float | bool | str]:
        """
        Collect scalar hyperparameters from this instance for logging.

        Inspects all instance attributes and returns those whose value is a
        plain ``int``, ``float``, ``bool``, or ``str``.
        This is used by :meth:`save_metadata` and passed directly to MLflow via
        :meth:`~corl.trainer.algorithm.discrete.monte_carlo.TrainerMonteCarlo.params`.

        Returns:
            dict[str, int | float | bool | str]: Mapping of attribute name to
            scalar value for every qualifying instance attribute.
        """

        metadata = {}

        for name, value in self.__dict__.items():
            if type(value) in (
                int,
                float,
                bool,
                str,
            ):
                metadata[name] = value

        return metadata

    def save_metadata(self, model_dir: str) -> None:
        """
        Write :meth:`metadata` to ``<model_dir>/metadata.json``.

        Args:
            model_dir: Target directory for ``metadata.json``.

        Raises:
            FileNotFoundError: If ``model_dir`` does not exist.
            OSError: If the file cannot be written.
        """
        with open(Path(model_dir) / "metadata.json", "w") as f:
            json.dump(self.metadata(), f, indent=4)

    @abstractmethod
    def load_metadata(self, model_dir: str) -> None:
        """
        Load :meth:`metadata` from ``<model_dir>/metadata.json``.

        Args:
            model_dir: Source directory containing ``metadata.json``.

        Raises:
            FileNotFoundError: If ``metadata.json`` does not exist in
                ``model_dir``.
            json.JSONDecodeError: If the file is not valid JSON.
        """

    def save(self, model_dir: str) -> None:
        """
        Save weights and metadata to ``model_dir``.

        Convenience method that calls :meth:`save_weights` followed by
        :meth:`save_metadata`.

        Args:
            model_dir: Target directory for weights and ``metadata.json``.
        """
        self.save_weights(model_dir)
        self.save_metadata(model_dir)

    def load(self, model_dir: str) -> None:
        """
        Load weights and metadata from ``model_dir``.

        Convenience method that calls :meth:`load_weights` followed by
        :meth:`load_metadata`.

        Args:
            model_dir: Source directory containing weights and
                ``metadata.json``.
        """
        self.load_weights(model_dir)
        self.load_metadata(model_dir)
