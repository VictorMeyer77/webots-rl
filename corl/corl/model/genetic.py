import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from corl.model.model import Model

logger = logging.getLogger(__name__)


class ModelGenetic(Model):
    """
    Genetic-algorithm-based model that maps discrete observations to actions.

    The model is represented as a lookup table (``actions``) where the first
    element of each observation vector is used as an integer index to retrieve
    the corresponding action row.  Weights are stored as a plain NumPy ``.npy``
    file and can be persisted either as a final snapshot or as a versioned
    checkpoint.

    Attributes:
        actions: 2-D action table of shape ``(n_states, action_dim)``.
            ``None`` until set via :meth:`set_weights` or :meth:`load_weights`.
    """

    actions: NDArray[np.float32] | None = None

    def set_weights(self, actions: NDArray[np.float32]) -> None:
        """
        Directly assign a weight array to the model.

        Args:
            actions: 2-D array of shape ``(n_states, action_dim)`` that maps
                each discrete state index to an action vector.
        """
        self.actions = actions

    def load_weights(self, model_dir: str) -> None:
        """
        Load model weights from ``<model_dir>/model.npy``.

        Args:
            model_dir: Directory that contains the ``model.npy`` weight file.

        Raises:
            FileNotFoundError: If ``model.npy`` does not exist in
                ``model_dir``.
        """
        model_path = Path(model_dir) / "model.npy"
        self.actions = np.load(model_path)
        logger.info(f"Model loaded from {model_path}.")

    def save_weights(self, model_dir: str, checkpoint: bool = False) -> None:
        """
        Persist model weights to ``model_dir``.

        When ``checkpoint=False`` the weights are written to
        ``<model_dir>/model.npy``, overwriting any previous file.
        When ``checkpoint=True`` they are written to
        ``<model_dir>/model_ckt_<checkpoint_index>.npy`` and
        :attr:`checkpoint_index` is incremented so subsequent calls produce
        unique filenames.

        Args:
            model_dir: Target directory for the weight file.
            checkpoint: If ``True``, save as a versioned checkpoint instead
                of overwriting the latest snapshot.

        Raises:
            RuntimeError: If :attr:`actions` has not been set yet.
        """
        if self.actions is None:
            raise RuntimeError("Cannot save: model is None.")
        if checkpoint:
            model_path = Path(model_dir) / f"model_ckt_{self.checkpoint_index}.npy"
            self.checkpoint_index += 1
            logger.debug(f"Checkpoint saved: {model_path}")
        else:
            model_path = Path(model_dir) / "model.npy"
            logger.info(f"Final model saved: {model_path} and logged to MLflow.")

        np.save(model_path, self.actions)

    def predict(self, observation: NDArray[np.float32]) -> NDArray[np.int32]:
        """
        Look up and return the action for a given observation.

        The first element of ``observation`` is cast to ``int`` and used as an
        index into :attr:`actions` to retrieve the corresponding action row,
        which is then cast to ``np.int32``.

        Args:
            observation: 1-D sensor/state vector whose first element encodes
                the discrete state index.

        Returns:
            NDArray[np.int32]: Action vector for the observed state.

        Raises:
            RuntimeError: If :attr:`actions` has not been set yet.
            IndexError: If the state index derived from ``observation[0]`` is
                outside the range of :attr:`actions`.
        """
        if self.actions is None:
            raise RuntimeError("Cannot predict: model is None.")
        action = self.actions[int(observation[0])].astype(np.int32)
        logger.debug(f"Predicted action {action} for observation {observation}.")
        return action
