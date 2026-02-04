"""Unit tests for the Supervisor class."""

import pytest
from app.core.supervisor import Supervisor, SupervisorError


@pytest.fixture
def supervisor():
    """Provide a fresh supervisor instance for each test."""
    return Supervisor()


class TestAddTrain:
    """Tests for the add_train method."""

    def test_add_train_success(self, supervisor):
        """Test successfully adding a new training session."""
        supervisor.add_train("train_001")

        assert "train_001" in supervisor.training
        assert supervisor.training["train_001"] == {}

    def test_add_train_multiple(self, supervisor):
        """Test adding multiple training sessions."""
        train_ids = ["train_001", "train_002", "train_003"]

        for train_id in train_ids:
            supervisor.add_train(train_id)

        assert len(supervisor.training) == 3
        for train_id in train_ids:
            assert train_id in supervisor.training

    def test_add_train_duplicate(self, supervisor):
        """Test adding a duplicate training session raises error."""
        supervisor.add_train("train_001")

        with pytest.raises(SupervisorError, match="Train ID train_001 already exists"):
            supervisor.add_train("train_001")

    def test_add_train_empty_string(self, supervisor):
        """Test adding empty string as train_id raises error."""
        with pytest.raises(SupervisorError, match="Train ID cannot be empty"):
            supervisor.add_train("")

    def test_add_train_with_special_characters(self, supervisor):
        """Test adding training sessions with special characters."""
        train_ids = ["train-001", "train_002", "train.003", "train@004"]

        for train_id in train_ids:
            supervisor.add_train(train_id)

        assert len(supervisor.training) == 4


class TestAddWorker:
    """Tests for the add_worker method."""

    def test_add_worker_success(self, supervisor):
        """Test successfully adding a worker to a training session."""
        supervisor.add_train("train_001")

        worker_id = supervisor.add_worker("train_001")

        assert worker_id == 0
        assert supervisor.training["train_001"][0] == 0

    def test_add_worker_multiple(self, supervisor):
        """Test adding multiple workers to the same training session."""
        supervisor.add_train("train_001")

        worker_ids = []
        for _ in range(5):
            worker_id = supervisor.add_worker("train_001")
            worker_ids.append(worker_id)

        assert worker_ids == [0, 1, 2, 3, 4]
        assert len(supervisor.training["train_001"]) == 5

    def test_add_worker_nonexistent_train(self, supervisor):
        """Test adding a worker to non-existent training session raises error."""
        with pytest.raises(
            SupervisorError, match="Train ID nonexistent does not exist"
        ):
            supervisor.add_worker("nonexistent")

    def test_add_worker_incremental_ids(self, supervisor):
        """Test that worker IDs are assigned incrementally."""
        supervisor.add_train("train_001")

        id1 = supervisor.add_worker("train_001")
        id2 = supervisor.add_worker("train_001")
        id3 = supervisor.add_worker("train_001")

        assert id1 == 0
        assert id2 == 1
        assert id3 == 2

    def test_add_worker_multiple_trains(self, supervisor):
        """Test adding workers to different training sessions."""
        supervisor.add_train("train_A")
        supervisor.add_train("train_B")

        worker_a = supervisor.add_worker("train_A")
        worker_b = supervisor.add_worker("train_B")

        assert worker_a == 0
        assert worker_b == 0
        assert len(supervisor.training["train_A"]) == 1
        assert len(supervisor.training["train_B"]) == 1

    def test_add_worker_initial_episode_is_zero(self, supervisor):
        """Test that new workers start with episode_id 0."""
        supervisor.add_train("train_001")
        worker_id = supervisor.add_worker("train_001")

        assert supervisor.training["train_001"][worker_id] == 0


class TestGetEpisodeId:
    """Tests for the get_episode_id method."""

    def test_get_episode_id_success(self, supervisor):
        """Test successfully retrieving episode ID for a worker."""
        supervisor.add_train("train_001")
        worker_id = supervisor.add_worker("train_001")

        episode_id = supervisor.get_episode_id("train_001", worker_id)

        assert episode_id == 0

    def test_get_episode_id_nonexistent_train(self, supervisor):
        """Test getting episode ID for non-existent training session."""
        with pytest.raises(
            SupervisorError, match="Train ID nonexistent does not exist"
        ):
            supervisor.get_episode_id("nonexistent", 0)

    def test_get_episode_id_nonexistent_worker(self, supervisor):
        """Test getting episode ID for non-existent worker."""
        supervisor.add_train("train_001")

        with pytest.raises(
            SupervisorError, match="Worker ID 99 does not exist for Train ID train_001"
        ):
            supervisor.get_episode_id("train_001", 99)

    def test_get_episode_id_after_increment(self, supervisor):
        """Test getting episode ID after incrementing."""
        supervisor.add_train("train_001")
        worker_id = supervisor.add_worker("train_001")

        supervisor.increment_episode_id("train_001", worker_id)
        episode_id = supervisor.get_episode_id("train_001", worker_id)

        assert episode_id == 1

    def test_get_episode_id_multiple_workers(self, supervisor):
        """Test getting episode IDs for multiple workers."""
        supervisor.add_train("train_001")
        worker_0 = supervisor.add_worker("train_001")
        worker_1 = supervisor.add_worker("train_001")

        ep_0 = supervisor.get_episode_id("train_001", worker_0)
        ep_1 = supervisor.get_episode_id("train_001", worker_1)

        assert ep_0 == 0
        assert ep_1 == 0


class TestIncrementEpisodeId:
    """Tests for the increment_episode_id method."""

    def test_increment_episode_id_success(self, supervisor):
        """Test successfully incrementing episode ID."""
        supervisor.add_train("train_001")
        worker_id = supervisor.add_worker("train_001")

        supervisor.increment_episode_id("train_001", worker_id)

        assert supervisor.get_episode_id("train_001", worker_id) == 1

    def test_increment_episode_id_multiple_times(self, supervisor):
        """Test incrementing episode ID multiple times."""
        supervisor.add_train("train_001")
        worker_id = supervisor.add_worker("train_001")

        for i in range(10):
            supervisor.increment_episode_id("train_001", worker_id)
            assert supervisor.get_episode_id("train_001", worker_id) == i + 1

    def test_increment_episode_id_nonexistent_train(self, supervisor):
        """Test incrementing episode ID for non-existent training session."""
        with pytest.raises(
            SupervisorError, match="Train ID nonexistent does not exist"
        ):
            supervisor.increment_episode_id("nonexistent", 0)

    def test_increment_episode_id_nonexistent_worker(self, supervisor):
        """Test incrementing episode ID for non-existent worker."""
        supervisor.add_train("train_001")

        with pytest.raises(
            SupervisorError, match="Worker ID 99 does not exist for Train ID train_001"
        ):
            supervisor.increment_episode_id("train_001", 99)

    def test_increment_episode_id_isolation(self, supervisor):
        """Test that incrementing one worker doesn't affect others."""
        supervisor.add_train("train_001")
        worker_0 = supervisor.add_worker("train_001")
        worker_1 = supervisor.add_worker("train_001")

        for _ in range(5):
            supervisor.increment_episode_id("train_001", worker_0)

        assert supervisor.get_episode_id("train_001", worker_0) == 5
        assert supervisor.get_episode_id("train_001", worker_1) == 0

    def test_increment_episode_id_returns_none(self, supervisor):
        """Test that increment_episode_id returns None."""
        supervisor.add_train("train_001")
        worker_id = supervisor.add_worker("train_001")

        result = supervisor.increment_episode_id("train_001", worker_id)

        assert result is None


class TestTrainingProperty:
    """Tests for the training property."""

    def test_training_property_returns_dict(self, supervisor):
        """Test that training property returns a dictionary."""
        assert isinstance(supervisor.training, dict)

    def test_training_property_empty_initially(self, supervisor):
        """Test that training property is empty initially."""
        assert supervisor.training == {}

    def test_training_property_reflects_state(self, supervisor):
        """Test that training property reflects current state."""
        supervisor.add_train("train_001")
        supervisor.add_worker("train_001")

        assert "train_001" in supervisor.training
        assert 0 in supervisor.training["train_001"]


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self, supervisor):
        """Test complete workflow of creating train, adding workers, and tracking episodes."""
        # Create training session
        supervisor.add_train("exp_001")

        # Add workers
        worker_0 = supervisor.add_worker("exp_001")
        worker_1 = supervisor.add_worker("exp_001")

        assert worker_0 == 0
        assert worker_1 == 1

        # Check initial episode IDs
        assert supervisor.get_episode_id("exp_001", worker_0) == 0
        assert supervisor.get_episode_id("exp_001", worker_1) == 0

        # Increment episodes
        for _ in range(3):
            supervisor.increment_episode_id("exp_001", worker_0)

        for _ in range(7):
            supervisor.increment_episode_id("exp_001", worker_1)

        # Verify final state
        assert supervisor.get_episode_id("exp_001", worker_0) == 3
        assert supervisor.get_episode_id("exp_001", worker_1) == 7

    def test_multiple_independent_training_sessions(self, supervisor):
        """Test handling multiple independent training sessions."""
        # Create multiple training sessions
        supervisor.add_train("train_A")
        supervisor.add_train("train_B")

        # Add workers to each
        worker_a = supervisor.add_worker("train_A")
        worker_b = supervisor.add_worker("train_B")

        # Increment episodes for train_A
        for _ in range(5):
            supervisor.increment_episode_id("train_A", worker_a)

        # Verify train_B is unaffected
        assert supervisor.get_episode_id("train_B", worker_b) == 0

        # Verify train_A has correct count
        assert supervisor.get_episode_id("train_A", worker_a) == 5

    def test_large_scale_scenario(self, supervisor):
        """Test handling large numbers of training sessions and workers."""
        num_trains = 10
        workers_per_train = 20

        # Create training sessions and workers
        for train_idx in range(num_trains):
            train_id = f"train_{train_idx:03d}"
            supervisor.add_train(train_id)

            for _ in range(workers_per_train):
                supervisor.add_worker(train_id)

        # Verify structure
        assert len(supervisor.training) == num_trains
        for train_idx in range(num_trains):
            train_id = f"train_{train_idx:03d}"
            assert len(supervisor.training[train_id]) == workers_per_train

    def test_worker_lifecycle(self, supervisor):
        """Test complete lifecycle of a worker from creation to many episodes."""
        supervisor.add_train("train_001")
        worker_id = supervisor.add_worker("train_001")

        # Simulate many episodes
        for expected_episode in range(1, 101):
            supervisor.increment_episode_id("train_001", worker_id)
            actual_episode = supervisor.get_episode_id("train_001", worker_id)
            assert actual_episode == expected_episode
