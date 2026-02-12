import pytest
from unittest.mock import Mock, MagicMock
import numpy as np
import time

from corl.trainer.tracker import Tracker, StepKey, StepResult
from corl.api.wrapper import Wrapper


@pytest.fixture
def mock_wrapper():
    """Create a mock Wrapper for testing."""
    wrapper = Mock(spec=Wrapper)
    wrapper.get_workers.return_value = []
    return wrapper


@pytest.fixture
def tracker(mock_wrapper):
    """Create a Tracker instance with mock wrapper."""
    return Tracker(train_id="test_train_123", api=mock_wrapper)


@pytest.fixture
def sample_step_key():
    """Create a sample StepKey for testing."""
    return StepKey(worker_id=1, episode_id=0, step=0)


@pytest.fixture
def sample_step_result():
    """Create a sample StepResult for testing."""
    return StepResult(
        observation=np.array([1.0, 2.0, 3.0]), action=0, reward=1.0, done=False
    )


class TestStepKey:
    """Tests for StepKey dataclass."""

    def test_step_key_creation(self):
        """Test StepKey can be created with proper fields."""
        key = StepKey(worker_id=1, episode_id=2, step=3)
        assert key.worker_id == 1
        assert key.episode_id == 2
        assert key.step == 3

    def test_step_key_equality(self):
        """Test StepKey equality comparison."""
        key1 = StepKey(worker_id=1, episode_id=0, step=0)
        key2 = StepKey(worker_id=1, episode_id=0, step=0)
        key3 = StepKey(worker_id=2, episode_id=0, step=0)
        assert key1 == key2
        assert key1 != key3


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_step_result_creation(self):
        """Test StepResult can be created with proper fields."""
        result = StepResult(
            observation=np.array([1.0, 2.0]), action=1, reward=0.5, done=True
        )
        assert np.array_equal(result.observation, np.array([1.0, 2.0]))
        assert result.action == 1
        assert result.reward == 0.5
        assert result.done is True

    def test_step_result_non_terminal(self):
        """Test StepResult for non-terminal state."""
        result = StepResult(
            observation=np.array([0.0]), action=0, reward=0.0, done=False
        )
        assert result.done is False


class TestTrackerInitialization:
    """Tests for Tracker initialization."""

    def test_tracker_initialization(self, mock_wrapper):
        """Test Tracker initializes with correct attributes."""
        tracker = Tracker(train_id="test_123", api=mock_wrapper)
        assert tracker.train_id == "test_123"
        assert tracker.api == mock_wrapper
        assert tracker._workers == {}
        assert tracker._buffer == {}
        assert tracker.last_refresh == 0

    def test_tracker_workers_empty(self, tracker):
        """Test workers() returns empty list when no workers."""
        assert tracker.workers() == []

    def test_tracker_workers_with_data(self, tracker):
        """Test workers() returns list of StepKeys."""
        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=0)
        tracker._workers[2] = StepKey(worker_id=2, episode_id=1, step=5)
        workers = tracker.workers()
        assert len(workers) == 2
        assert any(w.worker_id == 1 for w in workers)
        assert any(w.worker_id == 2 for w in workers)


class TestTrackerRefresh:
    """Tests for Tracker refresh functionality."""

    def test_refresh_adds_new_worker(self, tracker, mock_wrapper):
        """Test refresh adds new active workers."""
        mock_wrapper.get_workers.return_value = [{"id": 1, "status": True}]

        tracker.refresh()

        assert 1 in tracker._workers
        assert tracker._workers[1].worker_id == 1
        assert tracker._workers[1].episode_id == 0
        assert tracker._workers[1].step == 0
        assert 1 in tracker._buffer

    def test_refresh_adds_multiple_workers(self, tracker, mock_wrapper):
        """Test refresh adds multiple workers."""
        mock_wrapper.get_workers.return_value = [
            {"id": 1, "status": True},
            {"id": 2, "status": True},
            {"id": 3, "status": True},
        ]

        tracker.refresh()

        assert len(tracker._workers) == 3
        assert all(i in tracker._workers for i in [1, 2, 3])
        assert all(i in tracker._buffer for i in [1, 2, 3])

    def test_refresh_removes_inactive_worker(self, tracker, mock_wrapper):
        """Test refresh removes workers with status=False."""
        # Add a worker first
        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=0)
        tracker._buffer[1] = MagicMock()

        # Mark worker as inactive
        mock_wrapper.get_workers.return_value = [{"id": 1, "status": False}]

        tracker.refresh()

        assert 1 not in tracker._workers
        assert 1 not in tracker._buffer

    def test_refresh_ignores_existing_active_worker(self, tracker, mock_wrapper):
        """Test refresh doesn't reset existing active workers."""
        # Add worker with some progress
        tracker._workers[1] = StepKey(worker_id=1, episode_id=5, step=10)
        tracker._buffer[1] = MagicMock()

        mock_wrapper.get_workers.return_value = [{"id": 1, "status": True}]

        tracker.refresh()

        # Should preserve existing state
        assert tracker._workers[1].episode_id == 5
        assert tracker._workers[1].step == 10

    def test_refresh_respects_rate_limit(self, tracker, mock_wrapper):
        """Test refresh only calls API after REFRESH_RATE seconds."""
        mock_wrapper.get_workers.return_value = []

        # First refresh should call API
        tracker.refresh()
        assert mock_wrapper.get_workers.call_count == 1

        # Immediate second refresh should not call API
        tracker.refresh()
        assert mock_wrapper.get_workers.call_count == 1

        # After time passes, should call API again
        tracker.last_refresh = time.time() - 10  # 10 seconds ago
        tracker.refresh()
        assert mock_wrapper.get_workers.call_count == 2

    def test_refresh_updates_last_refresh_time(self, tracker, mock_wrapper):
        """Test refresh updates last_refresh timestamp."""
        mock_wrapper.get_workers.return_value = []

        start_time = time.time()
        tracker.refresh()

        assert tracker.last_refresh >= start_time
        assert tracker.last_refresh <= time.time()


class TestTrackerIncrements:
    """Tests for Tracker increment functionality."""

    def test_increment_step(self, tracker):
        """Test increment_step increases step count."""
        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=0)

        tracker.increment_step(worker_id=1, episode_id=0)

        assert tracker._workers[1].step == 1

    def test_increment_step_multiple_times(self, tracker):
        """Test increment_step can be called multiple times."""
        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=0)

        tracker.increment_step(worker_id=1, episode_id=0)
        tracker.increment_step(worker_id=1, episode_id=0)
        tracker.increment_step(worker_id=1, episode_id=0)

        assert tracker._workers[1].step == 3

    def test_increment_step_invalid_worker(self, tracker):
        """Test increment_step raises error for unknown worker."""
        with pytest.raises(ValueError, match="Worker 999 not found"):
            tracker.increment_step(worker_id=999, episode_id=0)

    def test_increment_step_invalid_episode(self, tracker):
        """Test increment_step raises error for episode mismatch."""
        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=0)

        with pytest.raises(ValueError, match="Episode ID mismatch"):
            tracker.increment_step(worker_id=1, episode_id=5)

    def test_increment_episode(self, tracker):
        """Test increment_episode increases episode and resets step."""
        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=10)

        tracker.increment_episode(worker_id=1)

        assert tracker._workers[1].episode_id == 1
        assert tracker._workers[1].step == 0

    def test_increment_episode_multiple_times(self, tracker):
        """Test increment_episode can be called multiple times."""
        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=5)

        tracker.increment_episode(worker_id=1)
        tracker.increment_episode(worker_id=1)

        assert tracker._workers[1].episode_id == 2
        assert tracker._workers[1].step == 0

    def test_increment_episode_invalid_worker(self, tracker):
        """Test increment_episode raises error for unknown worker."""
        with pytest.raises(ValueError, match="Worker 999 not found"):
            tracker.increment_episode(worker_id=999)


class TestTrackerAddStepResult:
    """Tests for adding step results."""

    def test_add_step_result_valid(self, tracker, sample_step_key, sample_step_result):
        """Test add_step_result adds result to buffer."""
        from collections import deque

        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=0)
        tracker._buffer[1] = deque(maxlen=2)

        tracker.add_step_result(sample_step_key, sample_step_result)

        assert len(tracker._buffer[1]) == 1
        assert tracker._buffer[1][0] == (sample_step_key, sample_step_result)

    def test_add_step_result_multiple(
        self, tracker, sample_step_key, sample_step_result
    ):
        """Test add_step_result can add multiple results."""
        from collections import deque

        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=0)
        tracker._buffer[1] = deque(maxlen=2)

        key1 = StepKey(worker_id=1, episode_id=0, step=0)
        result1 = StepResult(np.array([1.0]), 0, 0.5, False)

        key2 = StepKey(worker_id=1, episode_id=0, step=0)
        result2 = StepResult(np.array([2.0]), 1, 1.0, False)

        tracker.add_step_result(key1, result1)
        tracker.add_step_result(key2, result2)

        assert len(tracker._buffer[1]) == 2

    def test_add_step_result_maxlen_overflow(self, tracker):
        """Test add_step_result respects deque maxlen=2."""
        from collections import deque

        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=0)
        tracker._buffer[1] = deque(maxlen=2)

        key1 = StepKey(worker_id=1, episode_id=0, step=0)
        result1 = StepResult(np.array([1.0]), 0, 0.5, False)

        key2 = StepKey(worker_id=1, episode_id=0, step=0)
        result2 = StepResult(np.array([2.0]), 1, 1.0, False)

        key3 = StepKey(worker_id=1, episode_id=0, step=0)
        result3 = StepResult(np.array([3.0]), 2, 1.5, False)

        tracker.add_step_result(key1, result1)
        tracker.add_step_result(key2, result2)
        tracker.add_step_result(key3, result3)

        # Should only keep last 2
        assert len(tracker._buffer[1]) == 2
        assert tracker._buffer[1][0] == (key2, result2)
        assert tracker._buffer[1][1] == (key3, result3)

    def test_add_step_result_invalid_worker(
        self, tracker, sample_step_key, sample_step_result
    ):
        """Test add_step_result raises error for unknown worker."""
        with pytest.raises(ValueError, match="Worker 1 not found"):
            tracker.add_step_result(sample_step_key, sample_step_result)

    def test_add_step_result_invalid_episode(self, tracker, sample_step_result):
        """Test add_step_result raises error for episode mismatch."""
        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=0)
        invalid_key = StepKey(worker_id=1, episode_id=5, step=0)

        with pytest.raises(ValueError, match="Episode ID mismatch"):
            tracker.add_step_result(invalid_key, sample_step_result)

    def test_add_step_result_invalid_step(self, tracker, sample_step_result):
        """Test add_step_result raises error for step mismatch."""
        from collections import deque

        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=0)
        tracker._buffer[1] = deque(maxlen=2)
        invalid_key = StepKey(worker_id=1, episode_id=0, step=5)

        with pytest.raises(ValueError, match="Step mismatch"):
            tracker.add_step_result(invalid_key, sample_step_result)


class TestTrackerGetStepResult:
    """Tests for retrieving step results."""

    def test_get_step_result_empty_buffer(self, tracker):
        """Test get_step_result returns empty list when buffer is empty."""
        from collections import deque

        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=0)
        tracker._buffer[1] = deque(maxlen=2)

        results = tracker.get_step_result()

        assert results == []

    def test_get_step_result_single_non_terminal(self, tracker):
        """Test get_step_result with single non-terminal step returns empty."""
        from collections import deque

        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=0)
        tracker._buffer[1] = deque(maxlen=2)

        key = StepKey(worker_id=1, episode_id=0, step=0)
        result = StepResult(np.array([1.0]), 0, 0.5, done=False)
        tracker._buffer[1].append((key, result))

        results = tracker.get_step_result()

        assert results == []
        assert len(tracker._buffer[1]) == 1  # Should not clear

    def test_get_step_result_single_terminal(self, tracker):
        """Test get_step_result with single terminal step."""
        from collections import deque

        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=0)
        tracker._buffer[1] = deque(maxlen=2)

        key = StepKey(worker_id=1, episode_id=0, step=0)
        result = StepResult(np.array([1.0]), 0, 1.0, done=True)
        tracker._buffer[1].append((key, result))

        results = tracker.get_step_result()

        assert len(results) == 1
        assert results[0] == (key, result, None)
        assert len(tracker._buffer[1]) == 0  # Should clear

    def test_get_step_result_two_non_terminal(self, tracker):
        """Test get_step_result with two non-terminal steps."""
        from collections import deque

        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=0)
        tracker._buffer[1] = deque(maxlen=2)

        key1 = StepKey(worker_id=1, episode_id=0, step=0)
        result1 = StepResult(np.array([1.0]), 0, 0.5, done=False)

        key2 = StepKey(worker_id=1, episode_id=0, step=1)
        result2 = StepResult(np.array([2.0]), 1, 0.7, done=False)

        tracker._buffer[1].append((key1, result1))
        tracker._buffer[1].append((key2, result2))

        results = tracker.get_step_result()

        assert len(results) == 1
        assert results[0] == (key1, result1, result2)
        assert len(tracker._buffer[1]) == 1  # Should keep most recent
        assert tracker._buffer[1][0] == (key2, result2)

    def test_get_step_result_two_steps_second_terminal(self, tracker):
        """Test get_step_result when second step is terminal."""
        from collections import deque

        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=0)
        tracker._buffer[1] = deque(maxlen=2)

        key1 = StepKey(worker_id=1, episode_id=0, step=0)
        result1 = StepResult(np.array([1.0]), 0, 0.5, done=False)

        key2 = StepKey(worker_id=1, episode_id=0, step=1)
        result2 = StepResult(np.array([2.0]), 1, 1.0, done=True)

        tracker._buffer[1].append((key1, result1))
        tracker._buffer[1].append((key2, result2))

        results = tracker.get_step_result()

        # Should return two results: transition and terminal
        assert len(results) == 2
        assert results[0] == (key1, result1, result2)
        assert results[1] == (key2, result2, None)
        assert len(tracker._buffer[1]) == 0  # Should clear

    def test_get_step_result_multiple_workers(self, tracker):
        """Test get_step_result with multiple workers."""
        from collections import deque

        # Worker 1
        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=0)
        tracker._buffer[1] = deque(maxlen=2)
        key1a = StepKey(worker_id=1, episode_id=0, step=0)
        result1a = StepResult(np.array([1.0]), 0, 0.5, done=False)
        key1b = StepKey(worker_id=1, episode_id=0, step=1)
        result1b = StepResult(np.array([2.0]), 1, 0.7, done=False)
        tracker._buffer[1].append((key1a, result1a))
        tracker._buffer[1].append((key1b, result1b))

        # Worker 2
        tracker._workers[2] = StepKey(worker_id=2, episode_id=0, step=0)
        tracker._buffer[2] = deque(maxlen=2)
        key2a = StepKey(worker_id=2, episode_id=0, step=0)
        result2a = StepResult(np.array([3.0]), 0, 0.3, done=False)
        key2b = StepKey(worker_id=2, episode_id=0, step=1)
        result2b = StepResult(np.array([4.0]), 1, 0.4, done=False)
        tracker._buffer[2].append((key2a, result2a))
        tracker._buffer[2].append((key2b, result2b))

        results = tracker.get_step_result()

        # Should get one result from each worker
        assert len(results) == 2
        worker_ids = [r[0].worker_id for r in results]
        assert 1 in worker_ids
        assert 2 in worker_ids

    def test_get_step_result_mixed_buffer_states(self, tracker):
        """Test get_step_result with workers in different buffer states."""
        from collections import deque

        # Worker 1: empty buffer
        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=0)
        tracker._buffer[1] = deque(maxlen=2)

        # Worker 2: one non-terminal step
        tracker._workers[2] = StepKey(worker_id=2, episode_id=0, step=0)
        tracker._buffer[2] = deque(maxlen=2)
        key2 = StepKey(worker_id=2, episode_id=0, step=0)
        result2 = StepResult(np.array([2.0]), 0, 0.5, done=False)
        tracker._buffer[2].append((key2, result2))

        # Worker 3: two non-terminal steps
        tracker._workers[3] = StepKey(worker_id=3, episode_id=0, step=0)
        tracker._buffer[3] = deque(maxlen=2)
        key3a = StepKey(worker_id=3, episode_id=0, step=0)
        result3a = StepResult(np.array([3.0]), 0, 0.3, done=False)
        key3b = StepKey(worker_id=3, episode_id=0, step=1)
        result3b = StepResult(np.array([4.0]), 1, 0.4, done=False)
        tracker._buffer[3].append((key3a, result3a))
        tracker._buffer[3].append((key3b, result3b))

        results = tracker.get_step_result()

        # Should only get result from worker 3
        assert len(results) == 1
        assert results[0][0].worker_id == 3


class TestTrackerValidation:
    """Tests for internal validation methods."""

    def test_validate_worker_id_valid(self, tracker):
        """Test _validate_worker_id passes for valid worker."""
        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=0)

        # Should not raise
        tracker._validate_worker_id(1)

    def test_validate_worker_id_invalid(self, tracker):
        """Test _validate_worker_id raises for invalid worker."""
        with pytest.raises(ValueError, match="Worker 999 not found"):
            tracker._validate_worker_id(999)

    def test_validate_episode_id_valid(self, tracker):
        """Test _validate_episode_id passes for matching episode."""
        tracker._workers[1] = StepKey(worker_id=1, episode_id=5, step=0)

        # Should not raise
        tracker._validate_episode_id(1, 5)

    def test_validate_episode_id_invalid(self, tracker):
        """Test _validate_episode_id raises for mismatched episode."""
        tracker._workers[1] = StepKey(worker_id=1, episode_id=5, step=0)

        with pytest.raises(ValueError, match="Episode ID mismatch"):
            tracker._validate_episode_id(1, 3)

    def test_validate_step_valid(self, tracker):
        """Test _validate_step passes for matching step."""
        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=10)

        # Should not raise
        tracker._validate_step(1, 10)

    def test_validate_step_invalid(self, tracker):
        """Test _validate_step raises for mismatched step."""
        tracker._workers[1] = StepKey(worker_id=1, episode_id=0, step=10)

        with pytest.raises(ValueError, match="Step mismatch"):
            tracker._validate_step(1, 5)


class TestTrackerIntegration:
    """Integration tests for complete workflows."""

    def test_full_episode_workflow(self, tracker, mock_wrapper):
        """Test complete episode from start to finish."""

        # Setup worker via refresh
        mock_wrapper.get_workers.return_value = [{"id": 1, "status": True}]
        tracker.refresh()

        # Step 0
        key0 = StepKey(worker_id=1, episode_id=0, step=0)
        result0 = StepResult(np.array([1.0]), 0, 0.1, done=False)
        tracker.add_step_result(key0, result0)
        tracker.increment_step(worker_id=1, episode_id=0)

        # No results yet (only 1 non-terminal)
        assert tracker.get_step_result() == []

        # Step 1
        key1 = StepKey(worker_id=1, episode_id=0, step=1)
        result1 = StepResult(np.array([2.0]), 1, 0.2, done=False)
        tracker.add_step_result(key1, result1)
        tracker.increment_step(worker_id=1, episode_id=0)

        # Should get first transition
        results = tracker.get_step_result()
        assert len(results) == 1
        assert results[0] == (key0, result0, result1)

        # Step 2 (terminal)
        key2 = StepKey(worker_id=1, episode_id=0, step=2)
        result2 = StepResult(np.array([3.0]), 2, 1.0, done=True)
        tracker.add_step_result(key2, result2)

        # Should get two results: transition + terminal
        results = tracker.get_step_result()
        assert len(results) == 2
        assert results[0] == (key1, result1, result2)
        assert results[1] == (key2, result2, None)

        # Buffer should be cleared
        assert len(tracker._buffer[1]) == 0

        # Increment episode
        tracker.increment_episode(worker_id=1)
        assert tracker._workers[1].episode_id == 1
        assert tracker._workers[1].step == 0

    def test_multiple_workers_parallel(self, tracker, mock_wrapper):
        """Test multiple workers processing steps in parallel."""

        # Setup two workers
        mock_wrapper.get_workers.return_value = [
            {"id": 1, "status": True},
            {"id": 2, "status": True},
        ]
        tracker.refresh()

        # Worker 1: step 0
        key1_0 = StepKey(worker_id=1, episode_id=0, step=0)
        result1_0 = StepResult(np.array([1.0]), 0, 0.1, done=False)
        tracker.add_step_result(key1_0, result1_0)
        tracker.increment_step(worker_id=1, episode_id=0)

        # Worker 2: step 0
        key2_0 = StepKey(worker_id=2, episode_id=0, step=0)
        result2_0 = StepResult(np.array([10.0]), 0, 0.5, done=False)
        tracker.add_step_result(key2_0, result2_0)
        tracker.increment_step(worker_id=2, episode_id=0)

        # Worker 1: step 1
        key1_1 = StepKey(worker_id=1, episode_id=0, step=1)
        result1_1 = StepResult(np.array([2.0]), 1, 0.2, done=False)
        tracker.add_step_result(key1_1, result1_1)
        tracker.increment_step(worker_id=1, episode_id=0)

        # Worker 2: step 1
        key2_1 = StepKey(worker_id=2, episode_id=0, step=1)
        result2_1 = StepResult(np.array([20.0]), 1, 0.6, done=False)
        tracker.add_step_result(key2_1, result2_1)
        tracker.increment_step(worker_id=2, episode_id=0)

        # Should get results from both workers
        results = tracker.get_step_result()
        assert len(results) == 2

        worker_ids = {r[0].worker_id for r in results}
        assert worker_ids == {1, 2}

    def test_worker_removal_clears_buffer(self, tracker, mock_wrapper):
        """Test that removing a worker clears its buffer."""

        # Add worker
        mock_wrapper.get_workers.return_value = [{"id": 1, "status": True}]
        tracker.refresh()

        # Add some data
        key = StepKey(worker_id=1, episode_id=0, step=0)
        result = StepResult(np.array([1.0]), 0, 0.5, done=False)
        tracker.add_step_result(key, result)

        assert 1 in tracker._buffer
        assert len(tracker._buffer[1]) == 1

        # Remove worker
        mock_wrapper.get_workers.return_value = [{"id": 1, "status": False}]
        tracker.last_refresh = 0  # Force refresh
        tracker.refresh()

        # Buffer should be gone
        assert 1 not in tracker._buffer
        assert 1 not in tracker._workers
