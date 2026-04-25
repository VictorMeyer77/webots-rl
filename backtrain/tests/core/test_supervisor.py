"""Unit tests for the Supervisor module.

This module contains comprehensive tests for the Supervisor class and SupervisorError
exception, covering all functionality including training session management, worker
registration, episode tracking, and error handling.
"""

import pytest
from app.core.supervisor import Supervisor, SupervisorError
from app.schemas.supervisor import TrainingSchema


class TestSupervisorError:
    """Test cases for SupervisorError exception."""

    def test_supervisor_error_is_exception(self):
        """Verify that SupervisorError is a proper Exception subclass."""
        error = SupervisorError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"


class TestSupervisorInit:
    """Test cases for Supervisor initialization."""

    def test_init_creates_empty_training_dict(self):
        """Verify that a new Supervisor starts with no training sessions."""
        supervisor = Supervisor()
        assert supervisor.trainings == []


class TestSupervisorAddTrain:
    """Test cases for adding training sessions."""

    def test_add_train_creates_new_session(self):
        """Verify that add_train creates a new training session."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        trainings = supervisor.trainings
        assert len(trainings) == 1
        assert trainings[0].id == "train_1"
        assert trainings[0].workers == []

    def test_add_train_multiple_sessions(self):
        """Verify that multiple training sessions can be created."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        supervisor.add_train("train_2")
        trainings = supervisor.trainings
        assert len(trainings) == 2
        train_ids = [t.id for t in trainings]
        assert "train_1" in train_ids
        assert "train_2" in train_ids

    def test_add_train_duplicate_raises_error(self):
        """Verify that adding a duplicate training ID raises SupervisorError."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        with pytest.raises(SupervisorError, match="Train ID train_1 already exists"):
            supervisor.add_train("train_1")

    def test_add_train_empty_id_raises_error(self):
        """Verify that adding an empty training ID raises SupervisorError."""
        supervisor = Supervisor()
        with pytest.raises(SupervisorError, match="Train ID cannot be empty"):
            supervisor.add_train("")


class TestSupervisorGetTrain:
    """Test cases for retrieving training sessions."""

    def test_get_train_returns_correct_schema(self):
        """Verify that get_train returns the correct training session."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        training = supervisor.get_train("train_1")
        assert isinstance(training, TrainingSchema)
        assert training.id == "train_1"
        assert training.workers == []

    def test_get_train_with_workers(self):
        """Verify that get_train includes worker information."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        worker_id = supervisor.add_worker("train_1")
        training = supervisor.get_train("train_1")
        assert len(training.workers) == 1
        assert training.workers[0].id == worker_id
        assert training.workers[0].episode_id == 0
        assert training.workers[0].status is True

    def test_get_train_nonexistent_raises_error(self):
        """Verify that getting a non-existent training session raises SupervisorError."""
        supervisor = Supervisor()
        with pytest.raises(SupervisorError, match="Train ID invalid does not exist"):
            supervisor.get_train("invalid")


class TestSupervisorAddWorker:
    """Test cases for adding workers to training sessions."""

    def test_add_worker_returns_zero_for_first_worker(self):
        """Verify that the first worker gets ID 0."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        worker_id = supervisor.add_worker("train_1")
        assert worker_id == 0

    def test_add_worker_increments_id(self):
        """Verify that worker IDs increment sequentially."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        worker_id_1 = supervisor.add_worker("train_1")
        worker_id_2 = supervisor.add_worker("train_1")
        worker_id_3 = supervisor.add_worker("train_1")
        assert worker_id_1 == 0
        assert worker_id_2 == 1
        assert worker_id_3 == 2

    def test_add_worker_initializes_episode_and_status(self):
        """Verify that new workers start with episode_id 0 and status True."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        worker_id = supervisor.add_worker("train_1")
        training = supervisor.get_train("train_1")
        worker = training.workers[0]
        assert worker.id == worker_id
        assert worker.episode_id == 0
        assert worker.status is True

    def test_add_worker_nonexistent_train_raises_error(self):
        """Verify that adding a worker to non-existent training raises SupervisorError."""
        supervisor = Supervisor()
        with pytest.raises(SupervisorError, match="Train ID invalid does not exist"):
            supervisor.add_worker("invalid")


class TestSupervisorGetWorker:
    """Test cases for retrieving individual workers."""

    def test_get_worker_returns_correct_schema(self):
        """Verify that get_worker returns the correct WorkerSchema."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        worker_id = supervisor.add_worker("train_1")
        worker = supervisor.get_worker("train_1", worker_id)
        assert worker.id == worker_id
        assert worker.episode_id == 0
        assert worker.status is True

    def test_get_worker_with_updated_episode_id(self):
        """Verify that get_worker reflects updated episode ID."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        worker_id = supervisor.add_worker("train_1")
        supervisor.increment_episode_id("train_1", worker_id)
        supervisor.increment_episode_id("train_1", worker_id)
        worker = supervisor.get_worker("train_1", worker_id)
        assert worker.episode_id == 2
        assert worker.status is True

    def test_get_worker_with_updated_status(self):
        """Verify that get_worker reflects updated status."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        worker_id = supervisor.add_worker("train_1")
        supervisor.update_worker_status("train_1", worker_id, False)
        worker = supervisor.get_worker("train_1", worker_id)
        assert worker.status is False
        assert worker.episode_id == 0

    def test_get_worker_with_updated_episode_and_status(self):
        """Verify that get_worker reflects both episode ID and status updates."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        worker_id = supervisor.add_worker("train_1")
        supervisor.increment_episode_id("train_1", worker_id)
        supervisor.increment_episode_id("train_1", worker_id)
        supervisor.increment_episode_id("train_1", worker_id)
        supervisor.update_worker_status("train_1", worker_id, False)
        worker = supervisor.get_worker("train_1", worker_id)
        assert worker.episode_id == 3
        assert worker.status is False

    def test_get_worker_multiple_workers_in_session(self):
        """Verify that get_worker retrieves the correct worker when multiple exist."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        worker_id_1 = supervisor.add_worker("train_1")
        worker_id_2 = supervisor.add_worker("train_1")
        worker_id_3 = supervisor.add_worker("train_1")

        # Update workers differently
        supervisor.increment_episode_id("train_1", worker_id_1)
        supervisor.increment_episode_id("train_1", worker_id_2)
        supervisor.increment_episode_id("train_1", worker_id_2)
        supervisor.update_worker_status("train_1", worker_id_3, False)

        # Verify each worker has correct state
        worker_1 = supervisor.get_worker("train_1", worker_id_1)
        assert worker_1.id == worker_id_1
        assert worker_1.episode_id == 1
        assert worker_1.status is True

        worker_2 = supervisor.get_worker("train_1", worker_id_2)
        assert worker_2.id == worker_id_2
        assert worker_2.episode_id == 2
        assert worker_2.status is True

        worker_3 = supervisor.get_worker("train_1", worker_id_3)
        assert worker_3.id == worker_id_3
        assert worker_3.episode_id == 0
        assert worker_3.status is False

    def test_get_worker_multiple_training_sessions(self):
        """Verify that get_worker retrieves workers from different training sessions."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        supervisor.add_train("train_2")

        worker_1 = supervisor.add_worker("train_1")
        worker_2 = supervisor.add_worker("train_2")

        supervisor.increment_episode_id("train_1", worker_1)
        supervisor.update_worker_status("train_2", worker_2, False)

        # Verify worker from train_1
        retrieved_worker_1 = supervisor.get_worker("train_1", worker_1)
        assert retrieved_worker_1.id == worker_1
        assert retrieved_worker_1.episode_id == 1
        assert retrieved_worker_1.status is True

        # Verify worker from train_2
        retrieved_worker_2 = supervisor.get_worker("train_2", worker_2)
        assert retrieved_worker_2.id == worker_2
        assert retrieved_worker_2.episode_id == 0
        assert retrieved_worker_2.status is False

    def test_get_worker_nonexistent_train_raises_error(self):
        """Verify that getting a worker from non-existent training raises SupervisorError."""
        supervisor = Supervisor()
        with pytest.raises(SupervisorError, match="Train ID invalid does not exist"):
            supervisor.get_worker("invalid", 0)

    def test_get_worker_nonexistent_worker_raises_error(self):
        """Verify that getting a non-existent worker raises SupervisorError."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        with pytest.raises(
            SupervisorError, match="Worker ID 99 does not exist for Train ID train_1"
        ):
            supervisor.get_worker("train_1", 99)

    def test_get_worker_after_status_toggle(self):
        """Verify that get_worker reflects status after multiple toggles."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        worker_id = supervisor.add_worker("train_1")

        supervisor.update_worker_status("train_1", worker_id, False)
        worker = supervisor.get_worker("train_1", worker_id)
        assert worker.status is False

        supervisor.update_worker_status("train_1", worker_id, True)
        worker = supervisor.get_worker("train_1", worker_id)
        assert worker.status is True

        supervisor.update_worker_status("train_1", worker_id, False)
        worker = supervisor.get_worker("train_1", worker_id)
        assert worker.status is False


class TestSupervisorIncrementEpisodeId:
    """Test cases for incrementing episode IDs."""

    def test_increment_episode_id_increases_counter(self):
        """Verify that increment_episode_id increases the counter by one."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        worker_id = supervisor.add_worker("train_1")
        supervisor.increment_episode_id("train_1", worker_id)
        episode_id = supervisor.get_worker("train_1", worker_id).episode_id
        assert episode_id == 1

    def test_increment_episode_id_multiple_times(self):
        """Verify that episode ID can be incremented multiple times."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        worker_id = supervisor.add_worker("train_1")
        for i in range(5):
            supervisor.increment_episode_id("train_1", worker_id)
        episode_id = supervisor.get_worker("train_1", worker_id).episode_id
        assert episode_id == 5

    def test_increment_episode_id_preserves_status(self):
        """Verify that incrementing episode ID preserves worker status."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        worker_id = supervisor.add_worker("train_1")
        supervisor.update_worker_status("train_1", worker_id, False)
        supervisor.increment_episode_id("train_1", worker_id)
        training = supervisor.get_train("train_1")
        assert training.workers[0].status is False

    def test_increment_episode_id_nonexistent_train_raises_error(self):
        """Verify that incrementing for non-existent training raises error."""
        supervisor = Supervisor()
        with pytest.raises(SupervisorError, match="Train ID invalid does not exist"):
            supervisor.increment_episode_id("invalid", 0)

    def test_increment_episode_id_nonexistent_worker_raises_error(self):
        """Verify that incrementing for non-existent worker raises error."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        with pytest.raises(SupervisorError, match="Worker ID 99 does not exist"):
            supervisor.increment_episode_id("train_1", 99)


class TestSupervisorUpdateWorkerStatus:
    """Test cases for updating worker status."""

    def test_update_worker_status_changes_status(self):
        """Verify that update_worker_status changes the worker's status."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        worker_id = supervisor.add_worker("train_1")
        supervisor.update_worker_status("train_1", worker_id, False)
        training = supervisor.get_train("train_1")
        assert training.workers[0].status is False

    def test_update_worker_status_preserves_episode_id(self):
        """Verify that updating status preserves episode counter."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        worker_id = supervisor.add_worker("train_1")
        supervisor.increment_episode_id("train_1", worker_id)
        supervisor.increment_episode_id("train_1", worker_id)
        supervisor.update_worker_status("train_1", worker_id, False)
        episode_id = supervisor.get_worker("train_1", worker_id).episode_id
        assert episode_id == 2

    def test_update_worker_status_toggle(self):
        """Verify that status can be toggled multiple times."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        worker_id = supervisor.add_worker("train_1")
        supervisor.update_worker_status("train_1", worker_id, False)
        supervisor.update_worker_status("train_1", worker_id, True)
        training = supervisor.get_train("train_1")
        assert training.workers[0].status is True

    def test_update_worker_status_nonexistent_train_raises_error(self):
        """Verify that updating status for non-existent training raises error."""
        supervisor = Supervisor()
        with pytest.raises(SupervisorError, match="Train ID invalid does not exist"):
            supervisor.update_worker_status("invalid", 0, False)

    def test_update_worker_status_nonexistent_worker_raises_error(self):
        """Verify that updating status for non-existent worker raises error."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        with pytest.raises(SupervisorError, match="Worker ID 99 does not exist"):
            supervisor.update_worker_status("train_1", 99, False)


class TestSupervisorTrainingsProperty:
    """Test cases for the trainings property."""

    def test_trainings_property_returns_list(self):
        """Verify that trainings property returns a list of training schemas."""
        supervisor = Supervisor()
        assert isinstance(supervisor.trainings, list)

    def test_trainings_property_with_multiple_sessions(self):
        """Verify that trainings property includes all training sessions."""
        supervisor = Supervisor()
        supervisor.add_train("train_1")
        supervisor.add_train("train_2")
        supervisor.add_worker("train_1")
        supervisor.add_worker("train_2")
        supervisor.add_worker("train_2")

        trainings = supervisor.trainings
        assert len(trainings) == 2
        train_dict = {t.id: t for t in trainings}
        assert len(train_dict["train_1"].workers) == 1
        assert len(train_dict["train_2"].workers) == 2


class TestSupervisorIntegration:
    """Integration tests for complex Supervisor workflows."""

    def test_complete_workflow(self):
        """Test a complete workflow with multiple sessions and workers."""
        supervisor = Supervisor()

        # Create two training sessions
        supervisor.add_train("train_1")
        supervisor.add_train("train_2")

        # Add workers to first session
        worker_1_1 = supervisor.add_worker("train_1")
        worker_1_2 = supervisor.add_worker("train_1")

        # Add worker to second session
        _ = supervisor.add_worker("train_2")

        # Progress episodes for workers
        supervisor.increment_episode_id("train_1", worker_1_1)
        supervisor.increment_episode_id("train_1", worker_1_1)
        supervisor.increment_episode_id("train_1", worker_1_2)

        # Update statuses
        supervisor.update_worker_status("train_1", worker_1_2, False)

        # Verify final state
        train_1 = supervisor.get_train("train_1")
        train_2 = supervisor.get_train("train_2")

        assert len(train_1.workers) == 2
        assert train_1.workers[0].episode_id == 2
        assert train_1.workers[0].status is True
        assert train_1.workers[1].episode_id == 1
        assert train_1.workers[1].status is False

        assert len(train_2.workers) == 1
        assert train_2.workers[0].episode_id == 0
        assert train_2.workers[0].status is True
