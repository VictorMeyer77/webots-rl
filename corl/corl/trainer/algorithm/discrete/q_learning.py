import logging

import numpy as np

from corl.schemas.tracker import Transition as TransitionSchema
from corl.trainer.algorithm.discrete.td_tabular import TrainerTDTabular

logger = logging.getLogger(__name__)


class TrainerQLearning(TrainerTDTabular):
    """
    Off-policy Q-learning trainer using a tabular Q-value function.

    Extends :class:`~corl.trainer.algorithm.discrete.td_tabular.TrainerTDTabular` with
    the Q-learning TD target:

    .. math::

        Q(s, a) \\leftarrow Q(s, a) + \\alpha \\bigl[r + \\gamma \\, \\max_{a'} Q(s', a') - Q(s, a)\\bigr]

    Unlike SARSA, the bootstrap term uses the *greedy* maximum over all
    next-state actions regardless of the behaviour policy, making this
    off-policy.
    """

    def update_value_table(self, transition: TransitionSchema) -> dict[str, float]:
        """
        Apply a single Q-learning update and return the TD error.

        For non-terminal transitions the TD target bootstraps from the maximum
        Q-value across all actions in the next state. For terminal transitions
        (``current_step.done=True``) the target is the immediate reward alone.

        Args:
            transition (TransitionSchema): A ``(current_step, next_step)``
                pair. ``next_step`` may be ``None`` only when
                ``current_step.done`` is ``True``.

        Returns:
            dict[str, float]: Metrics dict with keys:

            - ``"td_error"``: absolute TD error ``|td_target − Q(s, a)|``
              computed before the update is applied.
            - ``"reward"``: immediate reward from ``current_step``.
        """

        current, next_ = transition.current_step, transition.next_step

        obs_index = self.model.observation_to_index(current.observation)
        action_idx = int(current.action[0])
        if current.done:
            td_target = current.reward
        else:
            next_obs_index = self.model.observation_to_index(next_.observation)
            td_target = current.reward + self.gamma * np.max(
                self.model.value_table[next_obs_index]
            )
        q_old = self.model.value_table[obs_index][action_idx]
        td_error = td_target - q_old
        self.model.value_table[obs_index][action_idx] += self.alpha * td_error

        logger.debug(
            f"Q-update: obs={obs_index} action={action_idx} "
            f"reward={current.reward:.4f} td_target={td_target:.4f} "
            f"td_error={td_error:.4f} q_old={q_old:.4f} "
            f"done={current.done}"
        )

        return {
            "td_error": abs(td_error),
            "reward": current.reward if current.reward is not None else 0.0,
        }
