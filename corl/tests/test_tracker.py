"""
Unit tests for corl.trainer.tracker.Tracker

The Wrapper is fully mocked so no real HTTP server is required.
"""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from corl.api.wrapper import Wrapper
from corl.schemas.learning import Action, Environment
from corl.schemas.tracker import StepKey, StepResult
from corl.trainer.tracker import Tracker

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

TRAIN_ID = "train_001"
WORKER_TIMEOUT = 60  # mirrors DEFAULTS["TRAINER_WORKER_TIMEOUT"]


def _make_api() -> MagicMock:
    return MagicMock(spec=Wrapper)


def _make_tracker(workers: list[dict] | None = None) -> Tracker:
    """Return a Tracker pre-populated with *workers* (bypassing refresh)."""
    api = _make_api()
    tracker = Tracker(TRAIN_ID, WORKER_TIMEOUT, api)
    if workers:
        for w in workers:
            wid = w["worker_id"]
            tracker._workers[wid] = StepKey(
                worker_id=wid,
                episode_id=w.get("episode_id", 0),
                step=w.get("step", 0),
            )
            tracker._buffer_results[wid] = StepResult()
            tracker._worker_last_update[wid] = time.time()
    return tracker


def _obs() -> np.ndarray:
    return np.array([0.1, 0.2, 0.3], dtype=np.float32)


# ===========================================================================
# __init__
# ===========================================================================


class TestInit:
    def test_attributes_set(self):
        api = _make_api()
        t = Tracker(TRAIN_ID, WORKER_TIMEOUT, api)
        assert t.train_id == TRAIN_ID
        assert t.api is api
        assert t.worker_timeout == WORKER_TIMEOUT
        assert t._workers == {}
        assert t._buffer_results == {}
        assert t._worker_last_update == {}
        assert t._last_worker_refresh == 0


# ===========================================================================
# worker_step_keys
# ===========================================================================


class TestWorkerStepKeys:
    def test_empty_when_no_workers(self):
        t = _make_tracker()
        assert t.worker_step_keys() == []

    def test_returns_step_keys_for_all_workers(self):
        t = _make_tracker([{"worker_id": 0}, {"worker_id": 1}])
        keys = t.worker_step_keys()
        assert len(keys) == 2
        assert all(isinstance(k, StepKey) for k in keys)

    def test_returns_current_step_keys(self):
        t = _make_tracker([{"worker_id": 0, "episode_id": 2, "step": 5}])
        key = t.worker_step_keys()[0]
        assert key == StepKey(worker_id=0, episode_id=2, step=5)


# ===========================================================================
# refresh
# ===========================================================================


class TestRefresh:
    def test_does_not_call_api_before_refresh_interval(self):
        t = _make_tracker()
        t._last_worker_refresh = time.time()  # just refreshed
        t.refresh()
        t.api.get_workers.assert_not_called()

    def test_calls_api_when_interval_elapsed(self):
        t = _make_tracker()
        t._last_worker_refresh = 0
        t.api.get_workers.return_value = []
        t.refresh()
        t.api.get_workers.assert_called_once_with(TRAIN_ID)

    def test_adds_active_worker(self):
        t = _make_tracker()
        t._last_worker_refresh = 0
        t.api.get_workers.return_value = [{"id": "3", "status": True}]
        t.refresh()
        assert 3 in t._workers
        assert t._workers[3] == StepKey(worker_id=3, episode_id=0, step=0)

    def test_does_not_add_inactive_worker(self):
        t = _make_tracker()
        t._last_worker_refresh = 0
        t.api.get_workers.return_value = [{"id": "5", "status": False}]
        t.refresh()
        assert 5 not in t._workers

    def test_removes_worker_that_becomes_inactive(self):
        t = _make_tracker([{"worker_id": 2}])
        t._last_worker_refresh = 0
        t.api.get_workers.return_value = [{"id": "2", "status": False}]
        t.refresh()
        assert 2 not in t._workers

    def test_does_not_re_add_existing_active_worker(self):
        t = _make_tracker([{"worker_id": 1}])
        t._last_worker_refresh = 0
        t.api.get_workers.return_value = [{"id": "1", "status": True}]
        t.refresh()
        # still just one entry for worker 1
        assert list(t._workers.keys()) == [1]

    def test_updates_last_worker_refresh_timestamp(self):
        t = _make_tracker()
        t._last_worker_refresh = 0
        t.api.get_workers.return_value = []
        before = time.time()
        t.refresh()
        assert t._last_worker_refresh >= before

    def test_new_worker_buffer_is_reset(self):
        t = _make_tracker()
        t._last_worker_refresh = 0
        t.api.get_workers.return_value = [{"id": "7", "status": True}]
        t.refresh()
        assert 7 in t._buffer_results
        assert isinstance(t._buffer_results[7], StepResult)


# ===========================================================================
# increment_step
# ===========================================================================


class TestIncrementStep:
    def test_increments_step(self):
        t = _make_tracker([{"worker_id": 0, "episode_id": 1, "step": 3}])
        t.increment_step(0, 1)
        assert t._workers[0].step == 4

    def test_preserves_worker_and_episode_id(self):
        t = _make_tracker([{"worker_id": 0, "episode_id": 2, "step": 0}])
        t.increment_step(0, 2)
        assert t._workers[0].worker_id == 0
        assert t._workers[0].episode_id == 2

    def test_resets_buffer_after_increment(self):
        t = _make_tracker([{"worker_id": 0, "episode_id": 0, "step": 0}])
        t._buffer_results[0].action = 1
        t.increment_step(0, 0)
        assert t._buffer_results[0].action is None

    def test_raises_for_unknown_worker(self):
        t = _make_tracker()
        with pytest.raises(ValueError, match="not found"):
            t.increment_step(99, 0)

    def test_raises_for_wrong_episode_id(self):
        t = _make_tracker([{"worker_id": 0, "episode_id": 1, "step": 0}])
        with pytest.raises(ValueError, match="Episode ID mismatch"):
            t.increment_step(0, 99)


# ===========================================================================
# increment_episode
# ===========================================================================


class TestIncrementEpisode:
    def test_increments_episode_and_resets_step(self):
        t = _make_tracker([{"worker_id": 0, "episode_id": 2, "step": 10}])
        t.increment_episode(0)
        assert t._workers[0].episode_id == 3
        assert t._workers[0].step == 0

    def test_preserves_worker_id(self):
        t = _make_tracker([{"worker_id": 5, "episode_id": 0, "step": 0}])
        t.increment_episode(5)
        assert t._workers[5].worker_id == 5

    def test_resets_buffer_after_increment(self):
        t = _make_tracker([{"worker_id": 0, "episode_id": 0, "step": 0}])
        t._buffer_results[0].reward = 5.0
        t.increment_episode(0)
        assert t._buffer_results[0].reward is None

    def test_raises_for_unknown_worker(self):
        t = _make_tracker()
        with pytest.raises(ValueError, match="not found"):
            t.increment_episode(42)


# ===========================================================================
# get_buffer_none_observations
# ===========================================================================


class TestGetBufferNoneObservations:
    def test_returns_all_keys_when_no_observations(self):
        t = _make_tracker([{"worker_id": 0}, {"worker_id": 1}])
        keys = t.get_buffer_none_observations()
        assert len(keys) == 2

    def test_excludes_worker_with_observation(self):
        t = _make_tracker([{"worker_id": 0}, {"worker_id": 1}])
        t._buffer_results[0].observation = _obs()
        keys = t.get_buffer_none_observations()
        assert len(keys) == 1
        assert keys[0].worker_id == 1

    def test_returns_empty_when_all_observations_present(self):
        t = _make_tracker([{"worker_id": 0}, {"worker_id": 1}])
        t._buffer_results[0].observation = _obs()
        t._buffer_results[1].observation = _obs()
        assert t.get_buffer_none_observations() == []

    def test_returns_empty_when_no_workers(self):
        t = _make_tracker()
        assert t.get_buffer_none_observations() == []


# ===========================================================================
# get_buffer_none_environments
# ===========================================================================


class TestGetBufferNoneEnvironments:
    def test_returns_all_keys_when_no_rewards(self):
        t = _make_tracker([{"worker_id": 0}, {"worker_id": 1}])
        keys = t.get_buffer_none_environments()
        assert len(keys) == 2

    def test_excludes_worker_with_reward(self):
        t = _make_tracker([{"worker_id": 0}, {"worker_id": 1}])
        t._buffer_results[0].reward = 1.0
        keys = t.get_buffer_none_environments()
        assert len(keys) == 1
        assert keys[0].worker_id == 1

    def test_returns_empty_when_all_rewards_present(self):
        t = _make_tracker([{"worker_id": 0}])
        t._buffer_results[0].reward = 0.0
        assert t.get_buffer_none_environments() == []

    def test_returns_empty_when_no_workers(self):
        t = _make_tracker()
        assert t.get_buffer_none_environments() == []


# ===========================================================================
# add_buffer_actions
# ===========================================================================


class TestAddBufferActions:
    def test_stores_action_in_buffer(self):
        t = _make_tracker([{"worker_id": 0}])
        key = StepKey(worker_id=0, episode_id=0, step=0)
        t.add_buffer_actions([(key, Action(action=3))])
        assert t._buffer_results[0].action == 3

    def test_updates_last_update_timestamp(self):
        t = _make_tracker([{"worker_id": 0}])
        t._worker_last_update[0] = 0.0
        key = StepKey(worker_id=0, episode_id=0, step=0)
        t.add_buffer_actions([(key, Action(action=1))])
        assert t._worker_last_update[0] > 0.0

    def test_multiple_actions(self):
        t = _make_tracker([{"worker_id": 0}, {"worker_id": 1}])
        actions = [
            (StepKey(worker_id=0, episode_id=0, step=0), Action(action=1)),
            (StepKey(worker_id=1, episode_id=0, step=0), Action(action=2)),
        ]
        t.add_buffer_actions(actions)
        assert t._buffer_results[0].action == 1
        assert t._buffer_results[1].action == 2

    def test_raises_for_none_action(self):
        t = _make_tracker([{"worker_id": 0}])
        key = StepKey(worker_id=0, episode_id=0, step=0)
        with pytest.raises(ValueError, match="None action"):
            t.add_buffer_actions([(key, None)])

    def test_raises_for_unknown_worker(self):
        t = _make_tracker()
        key = StepKey(worker_id=99, episode_id=0, step=0)
        with pytest.raises(ValueError, match="not found"):
            t.add_buffer_actions([(key, Action(action=0))])


# ===========================================================================
# add_buffer_observations
# ===========================================================================


class TestAddBufferObservations:
    def test_stores_observation_in_buffer(self):
        t = _make_tracker([{"worker_id": 0}])
        obs = _obs()
        key = StepKey(worker_id=0, episode_id=0, step=0)
        t.add_buffer_observations([(key, obs)])
        assert np.array_equal(t._buffer_results[0].observation, obs)

    def test_updates_last_update_timestamp(self):
        t = _make_tracker([{"worker_id": 0}])
        t._worker_last_update[0] = 0.0
        key = StepKey(worker_id=0, episode_id=0, step=0)
        t.add_buffer_observations([(key, _obs())])
        assert t._worker_last_update[0] > 0.0

    def test_multiple_observations(self):
        t = _make_tracker([{"worker_id": 0}, {"worker_id": 1}])
        obs0, obs1 = _obs(), _obs() * 2
        pairs = [
            (StepKey(worker_id=0, episode_id=0, step=0), obs0),
            (StepKey(worker_id=1, episode_id=0, step=0), obs1),
        ]
        t.add_buffer_observations(pairs)
        assert np.array_equal(t._buffer_results[0].observation, obs0)
        assert np.array_equal(t._buffer_results[1].observation, obs1)

    def test_raises_for_none_observation(self):
        t = _make_tracker([{"worker_id": 0}])
        key = StepKey(worker_id=0, episode_id=0, step=0)
        with pytest.raises(ValueError, match="None observation"):
            t.add_buffer_observations([(key, None)])

    def test_raises_for_unknown_worker(self):
        t = _make_tracker()
        key = StepKey(worker_id=99, episode_id=0, step=0)
        with pytest.raises(ValueError, match="not found"):
            t.add_buffer_observations([(key, _obs())])


# ===========================================================================
# add_buffer_environments
# ===========================================================================


class TestAddBufferEnvironments:
    def test_stores_reward_and_done_in_buffer(self):
        t = _make_tracker([{"worker_id": 0}])
        key = StepKey(worker_id=0, episode_id=0, step=0)
        env = Environment(reward=5.0, done=True)
        t.add_buffer_environments([(key, env)])
        assert t._buffer_results[0].reward == 5.0
        assert t._buffer_results[0].done is True

    def test_updates_last_update_timestamp(self):
        t = _make_tracker([{"worker_id": 0}])
        t._worker_last_update[0] = 0.0
        key = StepKey(worker_id=0, episode_id=0, step=0)
        t.add_buffer_environments([(key, Environment(reward=1.0, done=False))])
        assert t._worker_last_update[0] > 0.0

    def test_multiple_environments(self):
        t = _make_tracker([{"worker_id": 0}, {"worker_id": 1}])
        pairs = [
            (
                StepKey(worker_id=0, episode_id=0, step=0),
                Environment(reward=1.0, done=False),
            ),
            (
                StepKey(worker_id=1, episode_id=0, step=0),
                Environment(reward=-1.0, done=True),
            ),
        ]
        t.add_buffer_environments(pairs)
        assert t._buffer_results[0].reward == 1.0
        assert t._buffer_results[1].reward == -1.0
        assert t._buffer_results[1].done is True

    def test_raises_for_none_environment(self):
        t = _make_tracker([{"worker_id": 0}])
        key = StepKey(worker_id=0, episode_id=0, step=0)
        with pytest.raises(ValueError, match="None environment"):
            t.add_buffer_environments([(key, None)])

    def test_raises_for_unknown_worker(self):
        t = _make_tracker()
        key = StepKey(worker_id=99, episode_id=0, step=0)
        with pytest.raises(ValueError, match="not found"):
            t.add_buffer_environments([(key, Environment())])

    def test_stores_zero_reward(self):
        t = _make_tracker([{"worker_id": 0}])
        key = StepKey(worker_id=0, episode_id=0, step=0)
        t.add_buffer_environments([(key, Environment(reward=0.0, done=False))])
        assert t._buffer_results[0].reward == 0.0

    def test_stores_negative_reward(self):
        t = _make_tracker([{"worker_id": 0}])
        key = StepKey(worker_id=0, episode_id=0, step=0)
        t.add_buffer_environments([(key, Environment(reward=-10.0, done=False))])
        assert t._buffer_results[0].reward == -10.0


# ===========================================================================
# get_buffered_step_results
# ===========================================================================


class TestGetBufferedStepResults:
    def _complete_buffer(self, tracker: Tracker, worker_id: int) -> None:
        tracker._buffer_results[worker_id].observation = _obs()
        tracker._buffer_results[worker_id].action = 1
        tracker._buffer_results[worker_id].reward = 1.0
        tracker._buffer_results[worker_id].done = False

    def test_returns_empty_when_no_workers(self):
        t = _make_tracker()
        assert t.get_buffered_step_results() == []

    def test_returns_empty_when_no_complete_buffers(self):
        t = _make_tracker([{"worker_id": 0}, {"worker_id": 1}])
        assert t.get_buffered_step_results() == []

    def test_returns_complete_results(self):
        t = _make_tracker([{"worker_id": 0}])
        self._complete_buffer(t, 0)
        results = t.get_buffered_step_results()
        assert len(results) == 1
        key, result = results[0]
        assert isinstance(key, StepKey)
        assert isinstance(result, StepResult)
        assert result.is_complete()

    def test_excludes_incomplete_buffers(self):
        t = _make_tracker([{"worker_id": 0}, {"worker_id": 1}])
        self._complete_buffer(t, 0)
        # worker 1 has no action
        t._buffer_results[1].observation = _obs()
        t._buffer_results[1].reward = 1.0
        t._buffer_results[1].done = False
        results = t.get_buffered_step_results()
        assert len(results) == 1
        assert results[0][0].worker_id == 0

    def test_returns_correct_step_key(self):
        t = _make_tracker([{"worker_id": 2, "episode_id": 3, "step": 7}])
        self._complete_buffer(t, 2)
        key, _ = t.get_buffered_step_results()[0]
        assert key == StepKey(worker_id=2, episode_id=3, step=7)

    def test_multiple_complete_workers(self):
        t = _make_tracker([{"worker_id": 0}, {"worker_id": 1}])
        self._complete_buffer(t, 0)
        self._complete_buffer(t, 1)
        assert len(t.get_buffered_step_results()) == 2


# ===========================================================================
# worker_timeouts
# ===========================================================================


class TestWorkerTimeouts:
    def test_removes_timed_out_worker(self):
        t = _make_tracker([{"worker_id": 0}])
        t._worker_last_update[0] = time.time() - WORKER_TIMEOUT - 1
        t.api.update_worker_status.return_value = None
        t.worker_timeouts()
        assert 0 not in t._workers

    def test_keeps_active_worker(self):
        t = _make_tracker([{"worker_id": 0}])
        t._worker_last_update[0] = time.time()
        t.worker_timeouts()
        assert 0 in t._workers

    def test_calls_api_to_mark_worker_inactive(self):
        t = _make_tracker([{"worker_id": 0}])
        t._worker_last_update[0] = time.time() - WORKER_TIMEOUT - 1
        t.worker_timeouts()
        t.api.update_worker_status.assert_called_once_with(TRAIN_ID, 0, False)

    def test_multiple_timeouts(self):
        t = _make_tracker([{"worker_id": 0}, {"worker_id": 1}, {"worker_id": 2}])
        t._worker_last_update[0] = time.time() - WORKER_TIMEOUT - 1
        t._worker_last_update[1] = time.time()
        t._worker_last_update[2] = time.time() - WORKER_TIMEOUT - 1
        t.worker_timeouts()
        assert 0 not in t._workers
        assert 1 in t._workers
        assert 2 not in t._workers


# ===========================================================================
# close_workers
# ===========================================================================


class TestCloseWorkers:
    def test_closes_specific_workers(self):
        t = _make_tracker([{"worker_id": 0}, {"worker_id": 1}])
        t.close_workers([0])
        assert 0 not in t._workers
        assert 1 in t._workers

    def test_closes_all_workers_when_none_given(self):
        t = _make_tracker([{"worker_id": 0}, {"worker_id": 1}])
        t.close_workers()
        assert t._workers == {}

    def test_calls_api_to_mark_inactive(self):
        t = _make_tracker([{"worker_id": 0}])
        t.close_workers([0])
        t.api.update_worker_status.assert_called_once_with(TRAIN_ID, 0, False)

    def test_cleans_up_buffer_and_timestamp(self):
        t = _make_tracker([{"worker_id": 0}])
        t.close_workers([0])
        assert 0 not in t._buffer_results
        assert 0 not in t._worker_last_update

    def test_continues_after_api_error(self):
        t = _make_tracker([{"worker_id": 0}, {"worker_id": 1}])
        t.api.update_worker_status.side_effect = Exception("network error")
        t.close_workers([0, 1])  # should not raise
        assert t._workers == {}

    def test_noop_for_empty_list(self):
        t = _make_tracker([{"worker_id": 0}])
        t.close_workers([])
        assert 0 in t._workers
        t.api.update_worker_status.assert_not_called()


# ===========================================================================
# _reset_buffer
# ===========================================================================


class TestResetBuffer:
    def test_creates_fresh_step_result(self):
        t = _make_tracker([{"worker_id": 0}])
        t._buffer_results[0].action = 5
        t._reset_buffer(0)
        assert t._buffer_results[0].action is None
        assert t._buffer_results[0].observation is None
        assert t._buffer_results[0].reward is None
        assert t._buffer_results[0].done is None

    def test_updates_last_update_timestamp(self):
        t = _make_tracker([{"worker_id": 0}])
        t._worker_last_update[0] = 0.0
        t._reset_buffer(0)
        assert t._worker_last_update[0] > 0.0


# ===========================================================================
# _validate_worker_id
# ===========================================================================


class TestValidateWorkerId:
    def test_passes_for_known_worker(self):
        t = _make_tracker([{"worker_id": 1}])
        t._validate_worker_id(1)  # no exception

    def test_raises_for_unknown_worker(self):
        t = _make_tracker()
        with pytest.raises(ValueError, match="Worker 7 not found"):
            t._validate_worker_id(7)


# ===========================================================================
# _validate_episode_id
# ===========================================================================


class TestValidateEpisodeId:
    def test_passes_for_correct_episode(self):
        t = _make_tracker([{"worker_id": 0, "episode_id": 3}])
        t._validate_episode_id(0, 3)  # no exception

    def test_raises_for_wrong_episode(self):
        t = _make_tracker([{"worker_id": 0, "episode_id": 3}])
        with pytest.raises(ValueError, match="Episode ID mismatch"):
            t._validate_episode_id(0, 99)


# ===========================================================================
# _add_none_buffer_error
# ===========================================================================


class TestAddNoneBufferError:
    def test_raises_value_error_with_data_type(self):
        key = StepKey(worker_id=0, episode_id=1, step=2)
        with pytest.raises(ValueError, match="None action"):
            Tracker._add_none_buffer_error(0, key, "action")

    def test_error_includes_episode_and_step(self):
        key = StepKey(worker_id=0, episode_id=4, step=7)
        with pytest.raises(ValueError, match="4") as exc_info:
            Tracker._add_none_buffer_error(0, key, "observation")
        assert "7" in str(exc_info.value)

    def test_error_includes_worker_id(self):
        key = StepKey(worker_id=5, episode_id=0, step=0)
        with pytest.raises(ValueError, match="5"):
            Tracker._add_none_buffer_error(5, key, "environment")
