import logging

from corl.schemas.tracker import StepResult
from corl.schemas.tracker import Transition as TransitionSchema

logger = logging.getLogger(__name__)


class Transition:
    """
    Stateful buffer that pairs consecutive steps into TD transitions.

    For each worker, remembers the previous :class:`~corl.schemas.tracker.StepResult`
    so that the next call to :meth:`make` can form a ``(current, next)`` pair
    suitable for a TD update. Episode boundaries (``done=True``) are handled by
    emitting an additional terminal transition ``(done_step, None)`` and clearing
    the worker's buffer.

    Attributes:
        _worker_last_result (dict[int, StepResult]): Maps each active worker ID
            to its most recent step result, held until the next step arrives.
    """

    def __init__(self):
        self._worker_last_result: dict[int, StepResult] = {}

    def make(self, worker_id: int, step_result: StepResult) -> list[TransitionSchema]:
        """
        Produce zero, one, or two transitions from a new step result.

        Behaviour by case:

        - **Mid-episode step** (not first, not done): emits one transition
          ``(prev, current)`` and stores ``current`` for the next call.
        - **Terminal step** (not first, done): emits ``(prev, current)`` and
          ``(current, None)``, then clears the worker buffer.
        - **First step of episode** (not done): stores the step and emits nothing.
        - **Immediate termination** (first step, done): emits ``(current, None)``
          with no predecessor; logs a warning as this is a degenerate episode.

        Args:
            worker_id (int): Identifier of the worker that produced the step.
            step_result (StepResult): The latest step result from that worker.

        Returns:
            list[TransitionSchema]: Zero, one, or two transitions ready for a
                TD update.
        """
        transitions: list[TransitionSchema] = []

        if worker_id in self._worker_last_result:
            if step_result.done:
                transitions = [
                    TransitionSchema(
                        current_step=self._worker_last_result.pop(worker_id),
                        next_step=step_result,
                    ),
                    TransitionSchema(current_step=step_result, next_step=None),
                ]
            else:
                transitions = [
                    TransitionSchema(
                        current_step=self._worker_last_result[worker_id],
                        next_step=step_result,
                    )
                ]
                self._worker_last_result[worker_id] = step_result
        else:
            if step_result.done:
                logger.warning(
                    f"Worker {worker_id} received a done step with no previous step. "
                    "This may indicate a degenerate episode (e.g. immediate timeout)."
                )
                transitions = [
                    TransitionSchema(current_step=step_result, next_step=None)
                ]
            else:
                self._worker_last_result[worker_id] = step_result

        return transitions
