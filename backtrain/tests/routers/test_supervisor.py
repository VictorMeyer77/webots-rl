"""Unit tests for the Supervisor Router."""

import pytest
from app.core.supervisor import Supervisor
from app.dependencies import get_supervisor
from app.routers.supervisor import router
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    """Create a FastAPI application with the supervisor router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def supervisor():
    """Provide a fresh supervisor instance for each test."""
    return Supervisor()


@pytest.fixture
def mock_get_supervisor(supervisor, monkeypatch, app):
    """Mock the get_supervisor dependency."""

    def _get_supervisor():
        return supervisor

    # Override the dependency in the app
    app.dependency_overrides[get_supervisor] = _get_supervisor

    yield supervisor

    # Clean up after test
    app.dependency_overrides.clear()


def test_add_train_success(client, mock_get_supervisor):
    """Test successfully creating a new training session."""
    payload = {"train_id": "train_001"}

    response = client.post("/supervisor/train", json=payload)

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"


def test_add_train_duplicate(client, mock_get_supervisor):
    """Test creating a training session with duplicate ID."""
    payload = {"train_id": "train_001"}

    # First creation should succeed
    response1 = client.post("/supervisor/train", json=payload)
    assert response1.status_code == 201

    # Second creation should fail
    response2 = client.post("/supervisor/train", json=payload)
    assert response2.status_code == 409
    assert "detail" in response2.json()


def test_add_train_multiple_sessions(client, mock_get_supervisor):
    """Test creating multiple training sessions with different IDs."""
    train_ids = ["train_001", "train_002", "train_003"]

    for train_id in train_ids:
        response = client.post("/supervisor/train", json={"train_id": train_id})
        assert response.status_code == 201


def test_add_train_missing_train_id(client, mock_get_supervisor):
    """Test validation error when train_id is missing."""
    payload = {}

    response = client.post("/supervisor/train", json=payload)

    assert response.status_code == 422


def test_add_train_empty_train_id(client, mock_get_supervisor):
    """Test validation error for empty train_id."""
    payload = {"train_id": ""}

    response = client.post("/supervisor/train", json=payload)

    assert response.status_code == 409


def test_del_train_success(client, mock_get_supervisor):
    """Test successfully deleting a training session."""
    client.post("/supervisor/train", json={"train_id": "train_001"})

    response = client.delete("/supervisor/train/train_001")

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_del_train_removes_session(client, mock_get_supervisor):
    """Test that deleted session is no longer accessible."""
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.delete("/supervisor/train/train_001")

    response = client.get("/supervisor/train/train_001")
    assert response.status_code == 404


def test_del_train_nonexistent_returns_404(client, mock_get_supervisor):
    """Test deleting a non-existent training session returns 404."""
    response = client.delete("/supervisor/train/nonexistent")

    assert response.status_code == 404
    assert "detail" in response.json()


def test_del_train_allows_readd(client, mock_get_supervisor):
    """Test that a deleted session ID can be re-created."""
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.delete("/supervisor/train/train_001")

    response = client.post("/supervisor/train", json={"train_id": "train_001"})
    assert response.status_code == 201


def test_add_worker_success(client, mock_get_supervisor):
    """Test successfully adding a worker to a training session."""
    # Create training session first
    client.post("/supervisor/train", json={"train_id": "train_001"})

    response = client.post("/supervisor/train/train_001/worker")

    assert response.status_code == 201
    data = response.json()
    assert "worker_id" in data
    assert data["worker_id"] == 0


def test_add_worker_multiple_workers(client, mock_get_supervisor):
    """Test adding multiple workers to the same training session."""
    # Create training session
    client.post("/supervisor/train", json={"train_id": "train_001"})

    # Add multiple workers
    worker_ids = []
    for _ in range(3):
        response = client.post("/supervisor/train/train_001/worker")
        assert response.status_code == 201
        worker_ids.append(response.json()["worker_id"])

    # Verify worker IDs are sequential
    assert worker_ids == [0, 1, 2]


def test_add_worker_nonexistent_train(client, mock_get_supervisor):
    """Test adding a worker to a non-existent training session."""
    response = client.post("/supervisor/train/nonexistent/worker")

    assert response.status_code == 404
    assert "detail" in response.json()


def test_increment_episode_id_success(client, mock_get_supervisor):
    """Test successfully incrementing episode ID for a worker."""
    # Setup: create train and worker
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")

    response = client.post("/supervisor/train/train_001/worker/0/episode/increment")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    # Verify episode ID was incremented
    get_response = client.get("/supervisor/train/train_001/worker/0")
    assert get_response.json()["episode_id"] == 1


def test_increment_episode_id_multiple_times(client, mock_get_supervisor):
    """Test incrementing episode ID multiple times."""
    # Setup
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")

    # Increment 5 times
    for _ in range(5):
        response = client.post("/supervisor/train/train_001/worker/0/episode/increment")
        assert response.status_code == 200

    # Verify final episode ID
    get_response = client.get("/supervisor/train/train_001/worker/0")
    assert get_response.json()["episode_id"] == 5


def test_increment_episode_id_nonexistent_train(client, mock_get_supervisor):
    """Test incrementing episode ID for non-existent training session."""
    response = client.post("/supervisor/train/nonexistent/worker/0/episode/increment")

    assert response.status_code == 404


def test_increment_episode_id_nonexistent_worker(client, mock_get_supervisor):
    """Test incrementing episode ID for non-existent worker."""
    # Create training session only
    client.post("/supervisor/train", json={"train_id": "train_001"})

    response = client.post("/supervisor/train/train_001/worker/99/episode/increment")

    assert response.status_code == 404


def test_complete_workflow(client, mock_get_supervisor):
    """Test complete workflow of creating train, adding workers, and tracking episodes."""
    # 1. Create training session
    train_response = client.post("/supervisor/train", json={"train_id": "exp_001"})
    assert train_response.status_code == 201

    # 2. Add workers
    worker1 = client.post("/supervisor/train/exp_001/worker")
    worker2 = client.post("/supervisor/train/exp_001/worker")
    assert worker1.json()["worker_id"] == 0
    assert worker2.json()["worker_id"] == 1

    # 3. Check initial episode IDs
    ep1 = client.get("/supervisor/train/exp_001/worker/0")
    ep2 = client.get("/supervisor/train/exp_001/worker/1")
    assert ep1.json()["episode_id"] == 0
    assert ep2.json()["episode_id"] == 0

    # 4. Increment worker 0's episodes
    for _ in range(3):
        client.post("/supervisor/train/exp_001/worker/0/episode/increment")

    # 5. Increment worker 1's episodes
    for _ in range(7):
        client.post("/supervisor/train/exp_001/worker/1/episode/increment")

    # 6. Verify final state
    ep1_final = client.get("/supervisor/train/exp_001/worker/0")
    ep2_final = client.get("/supervisor/train/exp_001/worker/1")
    assert ep1_final.json()["episode_id"] == 3
    assert ep2_final.json()["episode_id"] == 7


def test_parallel_training_sessions(client, mock_get_supervisor):
    """Test handling multiple independent training sessions."""
    # Create multiple training sessions
    client.post("/supervisor/train", json={"train_id": "train_A"})
    client.post("/supervisor/train", json={"train_id": "train_B"})

    # Add workers to each
    client.post("/supervisor/train/train_A/worker")
    client.post("/supervisor/train/train_B/worker")

    # Increment episodes for train_A
    for _ in range(5):
        client.post("/supervisor/train/train_A/worker/0/episode/increment")

    # Verify train_B is unaffected
    ep_b = client.get("/supervisor/train/train_B/worker/0")
    assert ep_b.json()["episode_id"] == 0

    # Verify train_A has correct count
    ep_a = client.get("/supervisor/train/train_A/worker/0")
    assert ep_a.json()["episode_id"] == 5


def test_train_id_with_special_characters(client, mock_get_supervisor):
    """Test train_id with special characters."""
    train_ids = ["train-001", "train_002", "train.003"]

    for train_id in train_ids:
        response = client.post("/supervisor/train", json={"train_id": train_id})
        assert response.status_code == 201

        # Verify worker can be added
        worker_response = client.post(f"/supervisor/train/{train_id}/worker")
        assert worker_response.status_code == 201


def test_large_worker_count(client, mock_get_supervisor):
    """Test handling a large number of workers."""
    client.post("/supervisor/train", json={"train_id": "train_001"})

    num_workers = 100
    for i in range(num_workers):
        response = client.post("/supervisor/train/train_001/worker")
        assert response.status_code == 201
        assert response.json()["worker_id"] == i


def test_episode_id_isolation_between_workers(client, mock_get_supervisor):
    """Test that episode IDs are isolated between different workers."""
    # Setup
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")
    client.post("/supervisor/train/train_001/worker")

    # Increment only worker 0
    for _ in range(10):
        client.post("/supervisor/train/train_001/worker/0/episode/increment")

    # Verify worker 1 is unaffected
    ep1 = client.get("/supervisor/train/train_001/worker/1")
    assert ep1.json()["episode_id"] == 0

    # Verify worker 0 has correct count
    ep0 = client.get("/supervisor/train/train_001/worker/0")
    assert ep0.json()["episode_id"] == 10


"""Unit tests for get_train and update_worker_status endpoints."""


def test_get_train_success(client, mock_get_supervisor):
    """Test successfully retrieving training session details."""
    # Setup: create training session and add workers
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")
    client.post("/supervisor/train/train_001/worker")

    response = client.get("/supervisor/train/train_001")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "train_001"
    assert "workers" in data
    assert len(data["workers"]) == 2
    assert data["workers"][0]["id"] == 0
    assert data["workers"][1]["id"] == 1


def test_get_train_with_worker_details(client, mock_get_supervisor):
    """Test retrieving training session with complete worker details."""
    # Setup: create training session and worker
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")

    # Increment episode for worker
    client.post("/supervisor/train/train_001/worker/0/episode/increment")
    client.post("/supervisor/train/train_001/worker/0/episode/increment")

    response = client.get("/supervisor/train/train_001")

    assert response.status_code == 200
    data = response.json()
    assert data["workers"][0]["id"] == 0
    assert data["workers"][0]["episode_id"] == 2
    assert data["workers"][0]["status"] is True


def test_get_train_nonexistent(client, mock_get_supervisor):
    """Test retrieving non-existent training session returns 404."""
    response = client.get("/supervisor/train/nonexistent_train")

    assert response.status_code == 404
    assert "detail" in response.json()
    assert "nonexistent_train" in response.json()["detail"]


def test_get_train_empty_workers(client, mock_get_supervisor):
    """Test retrieving training session with no workers."""
    # Create training session without workers
    client.post("/supervisor/train", json={"train_id": "train_001"})

    response = client.get("/supervisor/train/train_001")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "train_001"
    assert data["workers"] == []


def test_get_train_multiple_workers_different_episodes(client, mock_get_supervisor):
    """Test retrieving training session with workers at different episode counts."""
    # Setup: create training session and workers
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")
    client.post("/supervisor/train/train_001/worker")
    client.post("/supervisor/train/train_001/worker")

    # Increment episodes differently for each worker
    for _ in range(5):
        client.post("/supervisor/train/train_001/worker/0/episode/increment")

    for _ in range(3):
        client.post("/supervisor/train/train_001/worker/1/episode/increment")

    # Worker 2 stays at episode 0

    response = client.get("/supervisor/train/train_001")

    assert response.status_code == 200
    data = response.json()
    assert len(data["workers"]) == 3
    assert data["workers"][0]["episode_id"] == 5
    assert data["workers"][1]["episode_id"] == 3
    assert data["workers"][2]["episode_id"] == 0


def test_get_train_special_characters_in_id(client, mock_get_supervisor):
    """Test retrieving training session with special characters in ID."""
    train_id = "train-test_123.v2"
    client.post("/supervisor/train", json={"train_id": train_id})
    client.post(f"/supervisor/train/{train_id}/worker")

    response = client.get(f"/supervisor/train/{train_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == train_id


def test_get_train_after_status_updates(client, mock_get_supervisor):
    """Test retrieving training session after updating worker statuses."""
    # Setup
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")
    client.post("/supervisor/train/train_001/worker")

    # Update statuses using JSON body
    client.post(
        "/supervisor/train/train_001/worker/0/status", json={"worker_status": False}
    )
    client.post(
        "/supervisor/train/train_001/worker/1/status", json={"worker_status": True}
    )

    response = client.get("/supervisor/train/train_001")

    assert response.status_code == 200
    data = response.json()
    assert data["workers"][0]["status"] is False
    assert data["workers"][1]["status"] is True


def test_update_worker_status_success_to_false(client, mock_get_supervisor):
    """Test updating worker status from true to false."""
    mock_get_supervisor.add_train("train_001")
    mock_get_supervisor.add_worker("train_001")

    response = client.post(
        "/supervisor/train/train_001/worker/0/status", json={"worker_status": False}
    )

    assert response.status_code == 200
    assert response.json() == {"status": "success"}

    # Verify status was updated
    train_response = client.get("/supervisor/train/train_001")
    assert train_response.json()["workers"][0]["status"] is False


def test_update_worker_status_success_to_true(client, mock_get_supervisor):
    """Test updating worker status from false back to true."""
    mock_get_supervisor.add_train("train_001")
    mock_get_supervisor.add_worker("train_001")
    mock_get_supervisor.update_worker_status("train_001", 0, False)

    response = client.post(
        "/supervisor/train/train_001/worker/0/status", json={"worker_status": True}
    )

    assert response.status_code == 200
    assert response.json() == {"status": "success"}

    # Verify status was updated
    train_response = client.get("/supervisor/train/train_001")
    assert train_response.json()["workers"][0]["status"] is True


def test_update_worker_status_nonexistent_worker(client, mock_get_supervisor):
    """Test updating status for a non-existent worker."""
    mock_get_supervisor.add_train("train_001")

    response = client.post(
        "/supervisor/train/train_001/worker/99/status", json={"worker_status": False}
    )

    assert response.status_code == 404
    assert "worker" in response.json()["detail"].lower()
    assert "does not exist" in response.json()["detail"].lower()


def test_update_worker_status_multiple_workers(client, mock_get_supervisor):
    """Test updating status for multiple workers independently."""
    mock_get_supervisor.add_train("train_001")
    mock_get_supervisor.add_worker("train_001")
    mock_get_supervisor.add_worker("train_001")
    mock_get_supervisor.add_worker("train_001")

    # Update status for workers 0 and 2
    client.post(
        "/supervisor/train/train_001/worker/0/status", json={"worker_status": False}
    )
    client.post(
        "/supervisor/train/train_001/worker/2/status", json={"worker_status": False}
    )

    # Verify all workers have correct status
    train_response = client.get("/supervisor/train/train_001")
    workers = train_response.json()["workers"]
    assert workers[0]["status"] is False
    assert workers[1]["status"] is True  # Unchanged
    assert workers[2]["status"] is False


def test_update_worker_status_does_not_affect_episode(client, mock_get_supervisor):
    """Test that updating worker status doesn't affect episode counter."""
    mock_get_supervisor.add_train("train_001")
    mock_get_supervisor.add_worker("train_001")
    mock_get_supervisor.increment_episode_id("train_001", 0)
    mock_get_supervisor.increment_episode_id("train_001", 0)

    # Get initial episode ID
    initial_episode = client.get("/supervisor/train/train_001/worker/0")
    assert initial_episode.json()["episode_id"] == 2

    # Update status
    client.post(
        "/supervisor/train/train_001/worker/0/status", json={"worker_status": False}
    )

    # Verify episode ID unchanged
    after_episode = client.get("/supervisor/train/train_001/worker/0")
    assert after_episode.json()["episode_id"] == 2


def test_update_worker_status_toggle_multiple_times(client, mock_get_supervisor):
    """Test toggling worker status multiple times."""
    mock_get_supervisor.add_train("train_001")
    mock_get_supervisor.add_worker("train_001")

    # Toggle status 5 times
    for i in range(5):
        expected_status = i % 2 == 0  # False, True, False, True, False
        response = client.post(
            "/supervisor/train/train_001/worker/0/status",
            json={"worker_status": not expected_status},
        )
        assert response.status_code == 200

        # Verify status was updated
        train_response = client.get("/supervisor/train/train_001")
        assert train_response.json()["workers"][0]["status"] is (not expected_status)


def test_update_worker_status_different_training_sessions(client, mock_get_supervisor):
    """Test that worker status updates are isolated per training session."""
    mock_get_supervisor.add_train("train_A")
    mock_get_supervisor.add_train("train_B")
    mock_get_supervisor.add_worker("train_A")
    mock_get_supervisor.add_worker("train_B")

    # Update status only for train_A worker
    client.post(
        "/supervisor/train/train_A/worker/0/status", json={"worker_status": False}
    )

    # Verify train_A worker status changed
    train_a = client.get("/supervisor/train/train_A")
    assert train_a.json()["workers"][0]["status"] is False

    # Verify train_B worker status unchanged
    train_b = client.get("/supervisor/train/train_B")
    assert train_b.json()["workers"][0]["status"] is True


def test_update_worker_status_invalid_json(client, mock_get_supervisor):
    """Test updating worker status with invalid JSON payload."""
    mock_get_supervisor.add_train("train_001")
    mock_get_supervisor.add_worker("train_001")

    response = client.post(
        "/supervisor/train/train_001/worker/0/status", json={"invalid_field": False}
    )

    assert response.status_code == 422  # Validation error


def test_complete_workflow_with_status_updates(client, mock_get_supervisor):
    """Test complete workflow including status updates."""
    # 1. Create training session
    client.post("/supervisor/train", json={"train_id": "exp_001"})

    # 2. Add workers
    client.post("/supervisor/train/exp_001/worker")
    client.post("/supervisor/train/exp_001/worker")

    # 3. Get initial state
    initial_state = client.get("/supervisor/train/exp_001")
    assert len(initial_state.json()["workers"]) == 2
    assert all(w["status"] is True for w in initial_state.json()["workers"])

    # 4. Increment episodes and update statuses
    for _ in range(3):
        client.post("/supervisor/train/exp_001/worker/0/episode/increment")
    client.post(
        "/supervisor/train/exp_001/worker/0/status", json={"worker_status": False}
    )

    for _ in range(7):
        client.post("/supervisor/train/exp_001/worker/1/episode/increment")
    # Worker 1 remains active

    # 5. Verify final state
    final_state = client.get("/supervisor/train/exp_001")
    workers = final_state.json()["workers"]

    assert workers[0]["episode_id"] == 3
    assert workers[0]["status"] is False

    assert workers[1]["episode_id"] == 7
    assert workers[1]["status"] is True


"""Unit tests for get_worker endpoint."""


def test_get_worker_success(client, mock_get_supervisor):
    """Test successfully retrieving a single worker's details."""
    # Setup: create training session and worker
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")

    response = client.get("/supervisor/train/train_001/worker/0")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 0
    assert data["episode_id"] == 0
    assert data["status"] is True


def test_get_worker_with_incremented_episode(client, mock_get_supervisor):
    """Test retrieving worker with incremented episode counter."""
    # Setup: create training session and worker
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")

    # Increment episode multiple times
    for _ in range(5):
        client.post("/supervisor/train/train_001/worker/0/episode/increment")

    response = client.get("/supervisor/train/train_001/worker/0")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 0
    assert data["episode_id"] == 5
    assert data["status"] is True


def test_get_worker_with_status_false(client, mock_get_supervisor):
    """Test retrieving worker with inactive status."""
    # Setup: create training session and worker
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")

    # Update status to false
    client.post(
        "/supervisor/train/train_001/worker/0/status", json={"worker_status": False}
    )

    response = client.get("/supervisor/train/train_001/worker/0")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 0
    assert data["episode_id"] == 0
    assert data["status"] is False


def test_get_worker_nonexistent_train(client, mock_get_supervisor):
    """Test retrieving worker from non-existent training session."""
    response = client.get("/supervisor/train/nonexistent_train/worker/0")

    assert response.status_code == 404
    assert "detail" in response.json()
    assert "nonexistent_train" in response.json()["detail"].lower()


def test_get_worker_nonexistent_worker(client, mock_get_supervisor):
    """Test retrieving non-existent worker from existing training session."""
    # Setup: create training session without workers
    client.post("/supervisor/train", json={"train_id": "train_001"})

    response = client.get("/supervisor/train/train_001/worker/0")

    assert response.status_code == 404
    assert "detail" in response.json()
    assert "worker" in response.json()["detail"].lower()
    assert "does not exist" in response.json()["detail"].lower()


def test_get_worker_multiple_workers_isolation(client, mock_get_supervisor):
    """Test that getting one worker doesn't return data from other workers."""
    # Setup: create training session with multiple workers
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")  # Worker 0
    client.post("/supervisor/train/train_001/worker")  # Worker 1
    client.post("/supervisor/train/train_001/worker")  # Worker 2

    # Modify worker 0
    for _ in range(3):
        client.post("/supervisor/train/train_001/worker/0/episode/increment")
    client.post(
        "/supervisor/train/train_001/worker/0/status", json={"worker_status": False}
    )

    # Modify worker 2
    for _ in range(7):
        client.post("/supervisor/train/train_001/worker/2/episode/increment")

    # Get worker 1 - should be unaffected
    response = client.get("/supervisor/train/train_001/worker/1")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 1
    assert data["episode_id"] == 0  # Unaffected
    assert data["status"] is True  # Unaffected


def test_get_worker_after_status_toggle(client, mock_get_supervisor):
    """Test retrieving worker after multiple status changes."""
    # Setup: create training session and worker
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")

    # Toggle status multiple times
    client.post(
        "/supervisor/train/train_001/worker/0/status", json={"worker_status": False}
    )
    client.post(
        "/supervisor/train/train_001/worker/0/status", json={"worker_status": True}
    )
    client.post(
        "/supervisor/train/train_001/worker/0/status", json={"worker_status": False}
    )

    response = client.get("/supervisor/train/train_001/worker/0")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] is False  # Should reflect last update


def test_get_worker_with_special_train_id(client, mock_get_supervisor):
    """Test retrieving worker from training session with special characters in ID."""
    train_id = "train-test_123.v2"
    client.post("/supervisor/train", json={"train_id": train_id})
    client.post(f"/supervisor/train/{train_id}/worker")

    response = client.get(f"/supervisor/train/{train_id}/worker/0")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 0


def test_get_worker_high_episode_count(client, mock_get_supervisor):
    """Test retrieving worker with high episode counter."""
    # Setup: create training session and worker
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")

    # Increment episode many times
    num_episodes = 1000
    for _ in range(num_episodes):
        client.post("/supervisor/train/train_001/worker/0/episode/increment")

    response = client.get("/supervisor/train/train_001/worker/0")

    assert response.status_code == 200
    data = response.json()
    assert data["episode_id"] == num_episodes


def test_get_worker_different_training_sessions(client, mock_get_supervisor):
    """Test that workers with same ID in different training sessions are isolated."""
    # Setup: create multiple training sessions
    client.post("/supervisor/train", json={"train_id": "train_A"})
    client.post("/supervisor/train", json={"train_id": "train_B"})
    client.post("/supervisor/train/train_A/worker")  # Worker 0 in train_A
    client.post("/supervisor/train/train_B/worker")  # Worker 0 in train_B

    # Modify worker 0 in train_A only
    for _ in range(5):
        client.post("/supervisor/train/train_A/worker/0/episode/increment")
    client.post(
        "/supervisor/train/train_A/worker/0/status", json={"worker_status": False}
    )

    # Get worker 0 from train_A
    response_a = client.get("/supervisor/train/train_A/worker/0")
    assert response_a.status_code == 200
    data_a = response_a.json()
    assert data_a["episode_id"] == 5
    assert data_a["status"] is False

    # Get worker 0 from train_B - should be unaffected
    response_b = client.get("/supervisor/train/train_B/worker/0")
    assert response_b.status_code == 200
    data_b = response_b.json()
    assert data_b["episode_id"] == 0
    assert data_b["status"] is True


def test_get_worker_all_workers_in_session(client, mock_get_supervisor):
    """Test retrieving all workers individually from a training session."""
    # Setup: create training session with multiple workers
    client.post("/supervisor/train", json={"train_id": "train_001"})
    num_workers = 5
    for _ in range(num_workers):
        client.post("/supervisor/train/train_001/worker")

    # Retrieve each worker individually
    for worker_id in range(num_workers):
        response = client.get(f"/supervisor/train/train_001/worker/{worker_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == worker_id
        assert data["episode_id"] == 0
        assert data["status"] is True


def test_get_worker_complete_state_tracking(client, mock_get_supervisor):
    """Test that get_worker accurately reflects complete worker state over time."""
    # Setup
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")

    # Initial state
    response = client.get("/supervisor/train/train_001/worker/0")
    assert response.json() == {"id": 0, "episode_id": 0, "status": True}

    # After episodes
    for _ in range(3):
        client.post("/supervisor/train/train_001/worker/0/episode/increment")
    response = client.get("/supervisor/train/train_001/worker/0")
    assert response.json() == {"id": 0, "episode_id": 3, "status": True}

    # After status change
    client.post(
        "/supervisor/train/train_001/worker/0/status", json={"worker_status": False}
    )
    response = client.get("/supervisor/train/train_001/worker/0")
    assert response.json() == {"id": 0, "episode_id": 3, "status": False}

    # After more episodes
    for _ in range(2):
        client.post("/supervisor/train/train_001/worker/0/episode/increment")
    response = client.get("/supervisor/train/train_001/worker/0")
    assert response.json() == {"id": 0, "episode_id": 5, "status": False}


def test_get_worker_invalid_worker_id_negative(client, mock_get_supervisor):
    """Test retrieving worker with negative worker ID."""
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")

    response = client.get("/supervisor/train/train_001/worker/-1")

    assert response.status_code == 404
    assert "detail" in response.json()


def test_get_worker_worker_id_out_of_range(client, mock_get_supervisor):
    """Test retrieving worker with ID beyond existing workers."""
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")  # Only worker 0 exists
    client.post("/supervisor/train/train_001/worker")  # Worker 1 exists

    response = client.get("/supervisor/train/train_001/worker/10")

    assert response.status_code == 404
    assert "detail" in response.json()
