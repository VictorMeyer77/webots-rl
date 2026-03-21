"""
Unit tests for corl.memory.transition.Transition
"""

import logging

import numpy as np

from corl.memory.transition import Transition
from corl.schemas.tracker import StepResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(done: bool = False, reward: float = 1.0) -> StepResult:
    return StepResult(
        observation=np.zeros(4, dtype=np.float32),
        action=0,
        reward=reward,
        done=done,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTransitionInit:
    def test_worker_last_result_starts_empty(self):
        t = Transition()
        assert t._worker_last_result == {}


class TestFirstStep:
    def test_first_mid_episode_step_emits_nothing(self):
        t = Transition()
        assert t.make(worker_id=0, step_result=_step(done=False)) == []

    def test_first_step_is_buffered(self):
        t = Transition()
        step = _step(done=False)
        t.make(worker_id=0, step_result=step)
        assert t._worker_last_result[0] is step

    def test_first_step_done_emits_terminal_transition(self):
        t = Transition()
        step = _step(done=True)
        result = t.make(worker_id=0, step_result=step)
        assert len(result) == 1
        assert result[0].current_step is step
        assert result[0].next_step is None

    def test_first_step_done_logs_warning(self, caplog):
        t = Transition()
        with caplog.at_level(logging.WARNING, logger="corl.memory.transition"):
            t.make(worker_id=42, step_result=_step(done=True))
        assert any("42" in record.message for record in caplog.records)

    def test_first_step_done_does_not_buffer(self):
        t = Transition()
        t.make(worker_id=0, step_result=_step(done=True))
        assert 0 not in t._worker_last_result


class TestMidEpisodeStep:
    def test_second_step_emits_one_transition(self):
        t = Transition()
        t.make(worker_id=0, step_result=_step(done=False))
        result = t.make(worker_id=0, step_result=_step(done=False))
        assert len(result) == 1

    def test_transition_pairs_previous_and_current(self):
        t = Transition()
        s0, s1 = _step(), _step()
        t.make(worker_id=0, step_result=s0)
        result = t.make(worker_id=0, step_result=s1)
        assert result[0].current_step is s0
        assert result[0].next_step is s1

    def test_buffer_advances_to_current_step(self):
        t = Transition()
        s0, s1 = _step(), _step()
        t.make(worker_id=0, step_result=s0)
        t.make(worker_id=0, step_result=s1)
        assert t._worker_last_result[0] is s1

    def test_multiple_steps_chain_correctly(self):
        t = Transition()
        steps = [_step(reward=float(i)) for i in range(4)]
        t.make(worker_id=0, step_result=steps[0])
        for i in range(1, 4):
            result = t.make(worker_id=0, step_result=steps[i])
            assert result[0].current_step is steps[i - 1]
            assert result[0].next_step is steps[i]


class TestTerminalStep:
    def test_terminal_step_emits_two_transitions(self):
        t = Transition()
        t.make(worker_id=0, step_result=_step(done=False))
        result = t.make(worker_id=0, step_result=_step(done=True))
        assert len(result) == 2

    def test_terminal_first_transition_pairs_prev_and_current(self):
        t = Transition()
        s0, s1 = _step(done=False), _step(done=True)
        t.make(worker_id=0, step_result=s0)
        result = t.make(worker_id=0, step_result=s1)
        assert result[0].current_step is s0
        assert result[0].next_step is s1

    def test_terminal_second_transition_has_none_next(self):
        t = Transition()
        s1 = _step(done=True)
        t.make(worker_id=0, step_result=_step(done=False))
        result = t.make(worker_id=0, step_result=s1)
        assert result[1].current_step is s1
        assert result[1].next_step is None

    def test_terminal_step_clears_buffer(self):
        t = Transition()
        t.make(worker_id=0, step_result=_step(done=False))
        t.make(worker_id=0, step_result=_step(done=True))
        assert 0 not in t._worker_last_result

    def test_new_episode_can_start_after_terminal(self):
        t = Transition()
        t.make(worker_id=0, step_result=_step(done=False))
        t.make(worker_id=0, step_result=_step(done=True))
        result = t.make(worker_id=0, step_result=_step(done=False))
        assert result == []
        assert 0 in t._worker_last_result


class TestMultipleWorkers:
    def test_workers_are_tracked_independently(self):
        t = Transition()
        t.make(worker_id=0, step_result=_step())
        t.make(worker_id=1, step_result=_step())
        assert 0 in t._worker_last_result
        assert 1 in t._worker_last_result

    def test_terminal_only_clears_own_worker(self):
        t = Transition()
        t.make(worker_id=0, step_result=_step())
        t.make(worker_id=1, step_result=_step())
        t.make(worker_id=0, step_result=_step(done=True))
        assert 0 not in t._worker_last_result
        assert 1 in t._worker_last_result

    def test_each_worker_emits_correct_transitions(self):
        t = Transition()
        a0, a1 = _step(reward=10.0), _step(reward=20.0)
        b0, b1 = _step(reward=30.0), _step(reward=40.0)
        t.make(worker_id=0, step_result=a0)
        t.make(worker_id=1, step_result=b0)
        res_a = t.make(worker_id=0, step_result=a1)
        res_b = t.make(worker_id=1, step_result=b1)
        assert res_a[0].current_step is a0
        assert res_b[0].current_step is b0
