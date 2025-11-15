"""
Model abstraction for persistent artifacts (e.g., Q-tables, policies).

Defines a simple interface to standardize loading and saving named resources
under the base directory `MODEL_PATH`.

Responsibilities:
- Provide an abstract `load(name)` to retrieve a model artifact into memory.
- Provide an abstract `save(name)` to persist the current in-memory artifact.
- Encourage concrete subclasses (e.g., NumPy, Torch, JSON) to implement format-specific
  validation and error handling.

Notes:
- `MODEL_PATH` is currently hard-coded; consider environment variable, config file,
  or dependency injection for portability.
- Subclasses should document expected file extensions and atomic write strategy.
"""

from abc import ABC, abstractmethod

MODEL_PATH = "/Users/victormeyer/Dev/Self/webots-rl/output/model"


class Model(ABC):
    """
    Abstract base class for a persistable model artifact.

    Subclass Guidelines:
    - Maintain internal state (e.g., arrays, parameters) after `load`.
    - Implement `save` using a safe write pattern (temp file + rename if needed).
    - Raise domain-specific exceptions on failure (e.g., FileNotFoundError, custom).

    Args:
        (none) Construction is format-specific in subclasses.

    Attributes:
        (implementation-defined in subclasses)
    """

    @abstractmethod
    def load(self, name: str) -> None:
        """
        Load a named artifact from `MODEL_PATH`.

        Args:
            name (str): Logical model identifier (without or with extension, per subclass rules).

        Raises:
            FileNotFoundError: If the target file does not exist.
            RuntimeError: For format or integrity errors.
        """
        raise NotImplementedError("Method load() not implemented.")

    @abstractmethod
    def save(self, name: str) -> None:
        """
        Persist the current in-memory artifact to `MODEL_PATH`.

        Args:
            name (str): Logical model identifier used to construct the output path.

        Raises:
            RuntimeError: If serialization or filesystem write fails.
        """
        raise NotImplementedError("Method save() not implemented.")
