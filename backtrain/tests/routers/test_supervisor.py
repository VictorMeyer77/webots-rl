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
    assert "train_001" in data["message"]


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
    assert "worker 0" in data["message"]

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
