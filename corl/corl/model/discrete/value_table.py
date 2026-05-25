import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from corl.model.model import Model

logger = logging.getLogger(__name__)


class ModelValueTable(Model):
    """
    Tabular Q-value model for discrete observation and action spaces.

    Stores a 2-D value table of shape
    ``(observation_cardinality ** observation_size, action_size)`` where each
    entry ``value_table[s][a]`` holds the estimated return for taking action
    ``a`` in state ``s``. States are encoded as a flat integer index via
    :meth:`observation_to_index`.

    Can be constructed from scratch by supplying dimension parameters, or
    restored from disk by supplying ``model_dir``.

    Attributes:
        observation_cardinality: Number of discrete bins per observation
            dimension.
        observation_size: Number of observation dimensions (features).
        action_size: Number of discrete actions available to the agent.
        value_table: Q-value array of shape
            ``(observation_cardinality ** observation_size, action_size)``.
    """

    observation_cardinality: int
    observation_size: int
    action_size: int
    value_table: NDArray[np.float32]

    def __init__(
        self,
        observation_cardinality: int | None = None,
        observation_size: int | None = None,
        action_size: int | None = None,
        model_dir: str | None = None,
    ):
        """
        Initialise the value table, either from scratch or from a saved model.

        Exactly one of the two construction modes must be used:

        - **From scratch**: provide ``observation_cardinality``,
          ``observation_size``, and ``action_size``. A zero-initialised value
          table is created.
        - **From disk**: provide ``model_dir``. Weights and metadata are loaded
          via :meth:`load`.

        Args:
            observation_cardinality: Number of discrete bins per observation
                dimension. Required when ``model_dir`` is ``None``.
            observation_size: Number of observation dimensions. Required when
                ``model_dir`` is ``None``.
            action_size: Number of discrete actions. Required when
                ``model_dir`` is ``None``.
            model_dir: Directory containing a previously saved model. When
                provided, all dimension arguments are ignored. A warning is
                logged if dimension arguments are also supplied.

        Raises:
            ValueError: If ``model_dir`` is ``None`` and any dimension
                argument is missing.
        """
        super().__init__()

        if model_dir is not None:
            if any(
                p is not None
                for p in (observation_cardinality, observation_size, action_size)
            ):
                logger.warning(
                    "model_dir provided alongside dimension arguments; dimension arguments will be ignored."
                )
            self.load(model_dir)
        else:
            if (
                observation_cardinality is None
                or observation_size is None
                or action_size is None
            ):
                raise ValueError(
                    "observation_cardinality, observation_size, and action_size must be provided when model_dir is None."
                )

            self.observation_cardinality = observation_cardinality
            self.observation_size = observation_size
            self.action_size = action_size
            self.value_table = np.zeros(
                (observation_cardinality**observation_size, action_size),
                dtype=np.float32,
            )

    def load_weights(self, model_dir: str) -> None:
        """
        Load the value table from ``<model_dir>/model.npy``.

        Args:
            model_dir: Source directory containing ``model.npy``.

        Raises:
            FileNotFoundError: If ``model.npy`` does not exist in ``model_dir``.
        """
        model_path = Path(model_dir) / "model.npy"
        self.value_table = np.load(model_path, allow_pickle=False)
        logger.info(f"Model loaded from {model_path}.")

    def save_weights(self, model_dir: str) -> None:
        """
        Persist the value table to ``model_dir``.

        Args:
            model_dir: Target directory for the ``.npy`` weight file.
        """
        model_path = Path(model_dir) / "model.npy"
        np.save(model_path, self.value_table)
        logger.info(f"Model saved: {model_path}.")

    def load_metadata(self, model_dir: str) -> None:
        """
        Load dimension metadata from ``<model_dir>/metadata.json``.

        Reads ``observation_cardinality``, ``observation_size``, and
        ``action_size`` from the JSON file and sets them as instance
        attributes.

        Args:
            model_dir: Source directory containing ``metadata.json``.

        Raises:
            FileNotFoundError: If ``metadata.json`` does not exist in
                ``model_dir``.
            json.JSONDecodeError: If the file is not valid JSON.
            KeyError: If any required key is missing from the metadata.
        """
        with open(Path(model_dir) / "metadata.json", "r") as f:
            metadata = json.load(f)
        self.observation_cardinality = int(metadata["observation_cardinality"])
        self.observation_size = int(metadata["observation_size"])
        self.action_size = int(metadata["action_size"])
        logger.info(
            f"Loaded metadata: observation_cardinality={self.observation_cardinality}, observation_size={self.observation_size}, action_size={self.action_size}"
        )

    def predict(
        self, observation: NDArray[np.float32]
    ) -> NDArray[np.int32] | NDArray[np.float32]:
        """
        Return the greedy action for the given observation.

        Among all actions that share the maximum Q-value for this state,
        one is chosen uniformly at random to break ties.

        Args:
            observation: State vector from the environment.

        Returns:
            0-d integer array containing the action index with the highest
            estimated return.
        """
        values = self.value_table[self.observation_to_index(observation)]
        max_actions = np.flatnonzero(values == values.max())
        action = np.random.choice(max_actions)
        logger.debug(
            f"Taking best action {action} for state {observation} with Value {values.max()}"
        )
        return np.array(action, dtype=np.int32)

    def epsilon_greedy_policy(
        self, observation: NDArray[np.float32], epsilon: float
    ) -> int:
        """
        Select an action using an ε-greedy policy.

        With probability ``epsilon`` a random action is sampled uniformly
        (exploration); otherwise the greedy action from :meth:`predict` is
        returned (exploitation).

        Args:
            observation: State vector from the environment.
            epsilon: Exploration rate in ``[0, 1]``.

        Returns:
            int: Selected action index (Python ``int`` wrapping an
            ``np.int32`` value).
        """
        if np.random.random() < epsilon:
            action = np.random.randint(self.action_size)
            logger.debug(f"Taking random action {action} for state {observation}")
        else:
            action = self.predict(observation)
        return int(action)

    def observation_to_index(self, observation: NDArray[np.float32]) -> int:
        """
        Convert a discrete observation vector to a flat value-table row index.

        Each element of ``observation`` is treated as an integer bin in
        ``[0, observation_cardinality)``. The vector is mapped to a single
        integer using row-major (C-order) ravelling over a
        ``(observation_cardinality,) * observation_size`` grid.

        Args:
            observation: State vector whose elements are integer bin indices.

        Returns:
            Flat index into the first dimension of :attr:`value_table`.

        Raises:
            ValueError: If ``observation`` length does not match
                :attr:`observation_size`.
            ValueError: If any element of ``observation`` is outside
                ``[0, observation_cardinality)``.
        """
        obs_int = observation.astype(int)
        if len(obs_int) != self.observation_size:
            raise ValueError(
                f"Observation length {len(obs_int)} does not match "
                f"observation_size {self.observation_size}."
            )
        if np.any(obs_int < 0) or np.any(obs_int >= self.observation_cardinality):
            raise ValueError(
                f"Observation values must be in [0, {self.observation_cardinality}), "
                f"got {obs_int.tolist()}."
            )
        dims = (self.observation_cardinality,) * self.observation_size
        return int(np.ravel_multi_index(obs_int, dims))
