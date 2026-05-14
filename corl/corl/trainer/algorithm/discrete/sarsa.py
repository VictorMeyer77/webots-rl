import logging

from corl.schemas.tracker import Transition as TransitionSchema
from corl.trainer.algorithm.discrete.td_tabular import TrainerTDTabular

logger = logging.getLogger(__name__)


class TrainerSarsa(TrainerTDTabular):
    """
    On-policy SARSA trainer using a tabular Q-value function.

    Extends :class:`~corl.trainer.algorithm.discrete.td_tabular.TrainerTDTabular` with
    the SARSA (State–Action–Reward–State–Action) TD target:

    .. math::

        Q(s, a) \\leftarrow Q(s, a) + \\alpha \\bigl[r + \\gamma \\, Q(s', a') - Q(s, a)\\bigr]

    where :math:`a'` is the *actual* action taken in state :math:`s'` by the
    current policy — making this on-policy, in contrast to Q-learning which
    bootstraps from :math:`\\max_{a'} Q(s', a')`.
    """

    def update_value_table(self, transition: TransitionSchema) -> dict[str, float]:
        """
        Apply a single SARSA update and return the TD error.

        Computes the TD target using the action stored in ``next_step``
        (i.e. the action actually selected by the policy, not the greedy
        maximum). For terminal transitions (``current_step.done=True``) the
        target reduces to the immediate reward with no bootstrap term.

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
            next_obs_index, next_action = None, None
        else:
            next_obs_index = self.model.observation_to_index(next_.observation)
            next_action = int(next_.action[0])
            td_target = (
                current.reward
                + self.gamma * self.model.value_table[next_obs_index][next_action]
            )
        td_error = td_target - self.model.value_table[obs_index][action_idx]
        self.model.value_table[obs_index][action_idx] += self.alpha * td_error
        logger.debug(
            f"SARSA update: obs={obs_index} action={action_idx} "
            f"next_obs={next_obs_index} next_action={next_action} "
            f"td_target={td_target:.4f} td_error={td_error:.4f} "
            f"done={current.done}"
        )

        return {
            "td_error": abs(td_error),
            "reward": current.reward if current.reward is not None else 0.0,
        }
