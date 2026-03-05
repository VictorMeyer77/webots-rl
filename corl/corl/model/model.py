import json
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class Model(ABC):
    """
    Abstract base class for all RL models.

    Defines the interface for prediction, persistence, and metadata management
    that every concrete model must implement. Subclasses are expected to
    override :meth:`predict`, :meth:`save_weights`, and :meth:`load_weights`.

    Attributes:
        model_dir: Directory used for saving and loading weights and metadata.
            Declared here for type-checking purposes; must be set by the
            subclass before calling :meth:`save` or :meth:`load`.
        metadata: Arbitrary key-value store persisted alongside model weights
            as ``metadata.json``. Defaults to an empty dict — subclasses
            should initialise their own instance dict in ``__init__`` to avoid
            sharing state across instances.
        checkpoint_index: Counter incremented by subclasses each time a
            checkpoint is saved. Used to generate unique checkpoint filenames.
    """

    model_dir: str
    metadata: dict[str, Any] = {}
    checkpoint_index: int = 0

    @abstractmethod
    def predict(
        self, observation: NDArray[np.float32]
    ) -> NDArray[np.int32] | NDArray[np.float32]:
        """
        Compute an action given an observation.

        Args:
            observation: Sensor/state vector from the environment.

        Returns:
            NDArray[np.int32] | NDArray[np.float32]: Discrete or continuous
            action output, depending on the action space of the model.
        """

    @abstractmethod
    def save_weights(self, model_dir: str, checkpoint: bool = False) -> None:
        """
        Persist model weights to ``model_dir``.

        Args:
            model_dir: Target directory for the weight file(s).
            checkpoint: If ``True``, save as a versioned checkpoint (using
                :attr:`checkpoint_index`) rather than overwriting the latest
                weights in place.
        """

    @abstractmethod
    def load_weights(self, model_dir: str) -> None:
        """
        Restore model weights from ``model_dir``.

        Args:
            model_dir: Source directory containing the weight file(s).
        """

    def save_metadata(self, model_dir: str) -> None:
        """
        Write :attr:`metadata` to ``<model_dir>/metadata.json``.

        Args:
            model_dir: Target directory for ``metadata.json``.

        Raises:
            OSError: If the file cannot be written.
        """
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=4)

    def load_metadata(self, model_dir: str) -> None:
        """
        Load :attr:`metadata` from ``<model_dir>/metadata.json``.

        Args:
            model_dir: Source directory containing ``metadata.json``.

        Raises:
            FileNotFoundError: If ``metadata.json`` does not exist in
                ``model_dir``.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

    def save(self, model_dir: str) -> None:
        """
        Save weights and metadata to ``model_dir``.

        Convenience method that calls :meth:`save_weights` followed by
        :meth:`save_metadata`. Equivalent to a non-checkpoint save; to save
        a versioned checkpoint call :meth:`save_weights` directly with
        ``checkpoint=True``.

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

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        """
        Replace :attr:`metadata` with a new dict.

        Prefer this over direct assignment to ensure the instance receives
        its own dict rather than mutating the shared class-level default.

        Args:
            metadata: New metadata mapping to assign.
        """
        self.metadata = metadata
