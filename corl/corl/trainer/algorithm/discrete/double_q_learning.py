import logging

import numpy as np
from numpy.typing import NDArray

from corl.schemas.tracker import Transition as TransitionSchema
from corl.trainer.algorithm.discrete.td_tabular import TrainerTDTabular

logger = logging.getLogger(__name__)


class TrainerDoubleQLearning(TrainerTDTabular):
    """
    Double Q-learning trainer using two tabular Q-value functions.

    Extends :class:`~corl.trainer.algorithm.discrete.td_tabular.TrainerTDTabular` with
    the Double Q-learning update rule, which maintains two independent Q-tables
    (``Q_A`` stored in :attr:`~corl.model.value_table.ModelValueTable.value_table`
    and ``Q_B`` stored in :attr:`value_table_b`) to decorrelate action selection
    from action evaluation and reduce maximisation bias.

    On each transition, one table is chosen at random to update:

    .. math::

        a^* = \\arg\\max_{a'} Q_A(s', a')
        \\quad\\text{then}\\quad
        Q_A(s, a) \\leftarrow Q_A(s, a)
            + \\alpha \\bigl[r + \\gamma Q_B(s', a^*) - Q_A(s, a)\\bigr]

    and symmetrically with :math:`Q_A` and :math:`Q_B` swapped for the other
    half of updates. The ε-greedy policy uses the *sum* of both tables so that
    both estimates contribute to exploration decisions.

    Attributes:
        value_table_b: Second Q-value table (``Q_B``), same shape as
            :attr:`~corl.model.value_table.ModelValueTable.value_table`.
            Initialised to zeros and updated in parallel with ``Q_A``.
    """

    value_table_b: NDArray[np.float32]

    def __init__(self, *args, **kwargs):
        """
        Initialise the Double Q-learning trainer.

        All arguments are forwarded to
        :class:`~corl.trainer.algorithm.discrete.td_tabular.TrainerTDTabular`. After the
        parent is initialised, a second Q-table ``value_table_b`` is created
        with the same shape as the model's :attr:`value_table`, filled with
        zeros.
        """
        super().__init__(*args, **kwargs)
        self.value_table_b = np.zeros_like(self.model.value_table)

    def policy(self, observations: NDArray[np.float32]) -> NDArray[np.int32]:
        """
        Select actions using an ε-greedy policy over the combined Q-tables.

        The greedy action is chosen from ``Q_A + Q_B`` so that both estimates
        contribute equally to exploitation decisions.

        Args:
            observations: Array of shape ``(N, obs_dim)`` containing the
                current observations for ``N`` workers.

        Returns:
            Integer action array of shape ``(N,)``, one action per observation.
        """
        combined = self.model.value_table + self.value_table_b
        actions = []
        for observation in observations:
            if np.random.random() < self.epsilon:
                actions.append(np.random.randint(combined.shape[1]))
            else:
                obs_index = self.model.observation_to_index(observation)
                actions.append(int(np.argmax(combined[obs_index])))
        return np.array(actions, dtype=np.int32)

    def update_value_table(self, transition: TransitionSchema) -> dict[str, float]:
        """
        Apply a single Double Q-learning update and return metrics.

        Randomly selects which Q-table to update (50/50). The selected table
        chooses the greedy next action; the *other* table evaluates that action.
        For terminal transitions the bootstrap term is dropped.

        Args:
            transition: A ``(current_step, next_step)`` pair.
                ``next_step`` may be ``None`` only when
                ``current_step.done`` is ``True``.

        Returns:
            dict[str, float]: Metrics dict with keys:

            - ``"td_error"``: absolute TD error ``|td_target − Q(s, a)|``
              computed before the update is applied.
            - ``"reward"``: immediate reward from ``current_step``.
        """
        current, next_ = transition.current_step, transition.next_step
        obs_index = self.model.observation_to_index(current.observation)

        if np.random.random() < 0.5:
            q_update = self.model.value_table
            q_eval = self.value_table_b
        else:
            q_update = self.value_table_b
            q_eval = self.model.value_table

        if current.done:
            td_target = current.reward
        else:
            next_obs_index = self.model.observation_to_index(next_.observation)
            best_action = int(np.argmax(q_update[next_obs_index]))
            td_target = (
                current.reward + self.gamma * q_eval[next_obs_index][best_action]
            )  # evaluate

        action_idx = int(current.action[0])
        q_old = q_update[obs_index][action_idx]
        td_error = td_target - q_old
        q_update[obs_index][action_idx] += self.alpha * td_error

        logger.debug(
            f"Double-Q update: obs={obs_index} action={action_idx} "
            f"reward={current.reward:.4f} td_target={td_target:.4f} "
            f"td_error={td_error:.4f} q_old={q_old:.4f} "
            f"done={current.done}"
        )

        return {
            "td_error": abs(td_error),
            "reward": current.reward if current.reward is not None else 0.0,
        }
