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


def test_get_episode_id_success(client, mock_get_supervisor):
    """Test successfully retrieving episode ID for a worker."""
    # Setup: create train and worker
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")

    response = client.get("/supervisor/train/train_001/worker/0/episode")

    assert response.status_code == 200
    data = response.json()
    assert "episode_id" in data
    assert data["episode_id"] == 0


def test_get_episode_id_nonexistent_train(client, mock_get_supervisor):
    """Test getting episode ID for non-existent training session."""
    response = client.get("/supervisor/train/nonexistent/worker/0/episode")

    assert response.status_code == 404


def test_get_episode_id_nonexistent_worker(client, mock_get_supervisor):
    """Test getting episode ID for non-existent worker."""
    # Create training session only
    client.post("/supervisor/train", json={"train_id": "train_001"})

    response = client.get("/supervisor/train/train_001/worker/99/episode")

    assert response.status_code == 404


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
    get_response = client.get("/supervisor/train/train_001/worker/0/episode")
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
    get_response = client.get("/supervisor/train/train_001/worker/0/episode")
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
    ep1 = client.get("/supervisor/train/exp_001/worker/0/episode")
    ep2 = client.get("/supervisor/train/exp_001/worker/1/episode")
    assert ep1.json()["episode_id"] == 0
    assert ep2.json()["episode_id"] == 0

    # 4. Increment worker 0's episodes
    for _ in range(3):
        client.post("/supervisor/train/exp_001/worker/0/episode/increment")

    # 5. Increment worker 1's episodes
    for _ in range(7):
        client.post("/supervisor/train/exp_001/worker/1/episode/increment")

    # 6. Verify final state
    ep1_final = client.get("/supervisor/train/exp_001/worker/0/episode")
    ep2_final = client.get("/supervisor/train/exp_001/worker/1/episode")
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
    ep_b = client.get("/supervisor/train/train_B/worker/0/episode")
    assert ep_b.json()["episode_id"] == 0

    # Verify train_A has correct count
    ep_a = client.get("/supervisor/train/train_A/worker/0/episode")
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
    ep1 = client.get("/supervisor/train/train_001/worker/1/episode")
    assert ep1.json()["episode_id"] == 0

    # Verify worker 0 has correct count
    ep0 = client.get("/supervisor/train/train_001/worker/0/episode")
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

    # Update statuses
    client.post("/supervisor/train/train_001/worker/0/status?worker_status=false")
    client.post("/supervisor/train/train_001/worker/1/status?worker_status=true")

    response = client.get("/supervisor/train/train_001")

    assert response.status_code == 200
    data = response.json()
    assert data["workers"][0]["status"] is False
    assert data["workers"][1]["status"] is True


def test_update_worker_status_success_to_false(client, mock_get_supervisor):
    """Test successfully updating worker status from True to False."""
    # Setup
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")

    response = client.post(
        "/supervisor/train/train_001/worker/0/status?worker_status=false"
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    # Verify status changed
    train_info = client.get("/supervisor/train/train_001")
    assert train_info.json()["workers"][0]["status"] is False


def test_update_worker_status_success_to_true(client, mock_get_supervisor):
    """Test successfully updating worker status from False to True."""
    # Setup
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")

    # First set to false
    client.post("/supervisor/train/train_001/worker/0/status?worker_status=false")

    # Then set back to true
    response = client.post(
        "/supervisor/train/train_001/worker/0/status?worker_status=true"
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # Verify status changed back
    train_info = client.get("/supervisor/train/train_001")
    assert train_info.json()["workers"][0]["status"] is True


def test_update_worker_status_nonexistent_train(client, mock_get_supervisor):
    """Test updating worker status for non-existent training session."""
    response = client.post(
        "/supervisor/train/nonexistent/worker/0/status?worker_status=false"
    )

    assert response.status_code == 404
    assert "detail" in response.json()


def test_update_worker_status_nonexistent_worker(client, mock_get_supervisor):
    """Test updating status for non-existent worker."""
    # Create training session only
    client.post("/supervisor/train", json={"train_id": "train_001"})

    response = client.post(
        "/supervisor/train/train_001/worker/99/status?worker_status=false"
    )

    assert response.status_code == 404
    assert "detail" in response.json()
    assert "99" in response.json()["detail"]


def test_update_worker_status_multiple_workers(client, mock_get_supervisor):
    """Test updating status for multiple workers independently."""
    # Setup
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")
    client.post("/supervisor/train/train_001/worker")
    client.post("/supervisor/train/train_001/worker")

    # Update different workers to different statuses
    client.post("/supervisor/train/train_001/worker/0/status?worker_status=false")
    client.post("/supervisor/train/train_001/worker/2/status?worker_status=false")
    # Worker 1 remains True (default)

    # Verify statuses
    train_info = client.get("/supervisor/train/train_001")
    workers = train_info.json()["workers"]
    assert workers[0]["status"] is False
    assert workers[1]["status"] is True
    assert workers[2]["status"] is False


def test_update_worker_status_does_not_affect_episode(client, mock_get_supervisor):
    """Test that updating status doesn't affect episode ID."""
    # Setup
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")

    # Increment episodes
    for _ in range(5):
        client.post("/supervisor/train/train_001/worker/0/episode/increment")

    # Get episode before status update
    episode_before = client.get("/supervisor/train/train_001/worker/0/episode")

    # Update status
    client.post("/supervisor/train/train_001/worker/0/status?worker_status=false")

    # Verify episode unchanged
    episode_after = client.get("/supervisor/train/train_001/worker/0/episode")
    assert episode_before.json()["episode_id"] == episode_after.json()["episode_id"]
    assert episode_after.json()["episode_id"] == 5


def test_update_worker_status_toggle_multiple_times(client, mock_get_supervisor):
    """Test toggling worker status multiple times."""
    # Setup
    client.post("/supervisor/train", json={"train_id": "train_001"})
    client.post("/supervisor/train/train_001/worker")

    # Toggle status multiple times
    for i in range(5):
        expected_status = i % 2 == 0  # Alternates False, True, False, True, False
        response = client.post(
            f"/supervisor/train/train_001/worker/0/status?worker_status={str(not expected_status).lower()}"
        )
        assert response.status_code == 200

    # Verify final status
    train_info = client.get("/supervisor/train/train_001")
    # After 5 toggles starting from True: False, True, False, True, False
    assert train_info.json()["workers"][0]["status"] is False


def test_update_worker_status_different_training_sessions(client, mock_get_supervisor):
    """Test that status updates are isolated between training sessions."""
    # Setup two training sessions
    client.post("/supervisor/train", json={"train_id": "train_A"})
    client.post("/supervisor/train", json={"train_id": "train_B"})
    client.post("/supervisor/train/train_A/worker")
    client.post("/supervisor/train/train_B/worker")

    # Update status in train_A only
    client.post("/supervisor/train/train_A/worker/0/status?worker_status=false")

    # Verify train_A status changed
    train_a_info = client.get("/supervisor/train/train_A")
    assert train_a_info.json()["workers"][0]["status"] is False

    # Verify train_B status unchanged
    train_b_info = client.get("/supervisor/train/train_B")
    assert train_b_info.json()["workers"][0]["status"] is True


def test_update_worker_status_with_special_train_id(client, mock_get_supervisor):
    """Test updating worker status with special characters in train ID."""
    train_id = "train-test_123.v2"
    client.post("/supervisor/train", json={"train_id": train_id})
    client.post(f"/supervisor/train/{train_id}/worker")

    response = client.post(
        f"/supervisor/train/{train_id}/worker/0/status?worker_status=false"
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"


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
    client.post("/supervisor/train/exp_001/worker/0/status?worker_status=false")

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
