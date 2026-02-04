"""Unit tests for the Action Communication Router."""

import pytest
from app.core.memory import Memory
from app.dependencies import get_action_memory
from app.routers.action import router
from app.schemas import ActionSchema
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    """Create a FastAPI application with the action router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def action_memory():
    """Provide a fresh memory instance for each test."""
    return Memory(capacity=100)


@pytest.fixture
def mock_get_action_memory(action_memory, monkeypatch, app):
    """Mock the get_action_memory dependency."""

    def _get_memory():
        return action_memory

    # Override the dependency in the app
    app.dependency_overrides[get_action_memory] = _get_memory

    yield action_memory

    # Clean up after test
    app.dependency_overrides.clear()


def test_check_action_memory_health_empty(client, mock_get_action_memory):
    """Test health check returns correct stats for empty memory."""
    response = client.get("/action/health")

    assert response.status_code == 200
    data = response.json()
    assert "stats" in data
    assert data["stats"]["size"] == 0
    assert data["stats"]["capacity"] == 100
    assert data["stats"]["remaining"] == 100
    assert data["stats"]["usage_percent"] == 0.0


def test_check_action_memory_health_with_data(client, mock_get_action_memory):
    """Test health check returns correct stats when memory contains data."""
    memory = mock_get_action_memory

    # Add some actions
    for i in range(5):
        memory.add(("train_001", 0, 1, i), ActionSchema(action=i))

    response = client.get("/action/health")

    assert response.status_code == 200
    data = response.json()
    assert data["stats"]["size"] == 5
    assert data["stats"]["capacity"] == 100
    assert data["stats"]["remaining"] == 95
    assert data["stats"]["usage_percent"] == 5.0


def test_add_action_step_success(client, mock_get_action_memory):
    """Test successfully adding an action."""
    payload = {"action": 2}

    response = client.post("/action/train_001/0/1/10", json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "stored"}

    # Verify it was stored
    memory = mock_get_action_memory
    key = ("train_001", 0, 1, 10)
    assert key in memory


def test_add_action_step_multiple_actions(client, mock_get_action_memory):
    """Test adding multiple actions with different keys."""
    actions = [
        ("train_001", 0, 1, 0, 0),
        ("train_001", 0, 1, 1, 1),
        ("train_001", 0, 2, 0, 2),
        ("train_001", 1, 1, 0, 3),
    ]

    for train_id, worker_id, episode_id, step, action_value in actions:
        response = client.post(
            f"/action/{train_id}/{worker_id}/{episode_id}/{step}",
            json={"action": action_value},
        )
        assert response.status_code == 200

    memory = mock_get_action_memory
    assert len(memory) == 4


def test_add_action_step_invalid_action_type(client, mock_get_action_memory):
    """Test validation error when action is not an integer."""
    payload = {"action": "not_an_integer"}

    response = client.post("/action/train_001/0/1/10", json=payload)

    assert response.status_code == 422


def test_add_action_step_missing_action_field(client, mock_get_action_memory):
    """Test validation error when action field is missing."""
    payload = {}

    response = client.post("/action/train_001/0/1/10", json=payload)

    assert response.status_code == 422


def test_add_action_step_invalid_train_id(client, mock_get_action_memory):
    """Test validation error for empty train_id."""
    payload = {"action": 1}

    response = client.post("/action//0/1/10", json=payload)

    assert response.status_code == 404  # FastAPI returns 404 for missing path params


def test_add_action_step_negative_worker_id(client, mock_get_action_memory):
    """Test validation error for negative worker_id."""
    payload = {"action": 1}

    response = client.post("/action/train_001/-1/1/10", json=payload)

    assert response.status_code == 422


def test_add_action_step_negative_episode_id(client, mock_get_action_memory):
    """Test validation error for negative episode_id."""
    payload = {"action": 1}

    response = client.post("/action/train_001/0/-1/10", json=payload)

    assert response.status_code == 422


def test_add_action_step_negative_step(client, mock_get_action_memory):
    """Test validation error for negative step."""
    payload = {"action": 1}

    response = client.post("/action/train_001/0/1/-1", json=payload)

    assert response.status_code == 422


def test_get_action_step_success(client, mock_get_action_memory):
    """Test successfully retrieving an action."""
    memory = mock_get_action_memory
    key = ("train_001", 0, 1, 10)
    action = ActionSchema(action=3)
    memory.add(key, action)

    response = client.get("/action/train_001/0/1/10")

    assert response.status_code == 200
    data = response.json()
    assert data["action"] == 3


def test_get_action_step_not_found(client, mock_get_action_memory):
    """Test 404 error when action is not found."""
    response = client.get("/action/train_001/0/1/10")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "Action not found" in data["detail"]
    assert "train_id=train_001" in data["detail"]
    assert "worker_id=0" in data["detail"]
    assert "episode_id=1" in data["detail"]
    assert "step=10" in data["detail"]


def test_get_action_step_multiple_retrievals(client, mock_get_action_memory):
    """Test that get is non-destructive and action can be retrieved multiple times."""
    memory = mock_get_action_memory
    key = ("train_001", 0, 1, 10)
    action = ActionSchema(action=7)
    memory.add(key, action)

    # Retrieve multiple times
    response1 = client.get("/action/train_001/0/1/10")
    response2 = client.get("/action/train_001/0/1/10")
    response3 = client.get("/action/train_001/0/1/10")

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response3.status_code == 200
    assert response1.json()["action"] == 7
    assert response2.json()["action"] == 7
    assert response3.json()["action"] == 7


def test_get_action_step_different_keys(client, mock_get_action_memory):
    """Test retrieving actions with different hierarchical keys."""
    memory = mock_get_action_memory

    # Add actions with different keys
    memory.add(("train_001", 0, 1, 10), ActionSchema(action=1))
    memory.add(("train_001", 1, 1, 10), ActionSchema(action=2))
    memory.add(("train_001", 0, 2, 10), ActionSchema(action=3))
    memory.add(("train_002", 0, 1, 10), ActionSchema(action=4))

    # Retrieve each one
    assert client.get("/action/train_001/0/1/10").json()["action"] == 1
    assert client.get("/action/train_001/1/1/10").json()["action"] == 2
    assert client.get("/action/train_001/0/2/10").json()["action"] == 3
    assert client.get("/action/train_002/0/1/10").json()["action"] == 4


def test_add_and_get_action_workflow(client, mock_get_action_memory):
    """Test complete workflow of adding and retrieving an action."""
    # Add action
    add_response = client.post("/action/exp_001/2/5/100", json={"action": 42})
    assert add_response.status_code == 200
    assert add_response.json()["status"] == "stored"

    # Retrieve action
    get_response = client.get("/action/exp_001/2/5/100")
    assert get_response.status_code == 200
    assert get_response.json()["action"] == 42


def test_action_with_zero_values(client, mock_get_action_memory):
    """Test handling actions with zero as a valid action value."""
    response = client.post("/action/train_001/0/0/0", json={"action": 0})
    assert response.status_code == 200

    get_response = client.get("/action/train_001/0/0/0")
    assert get_response.status_code == 200
    assert get_response.json()["action"] == 0


def test_action_with_large_values(client, mock_get_action_memory):
    """Test handling actions with large integer values."""
    large_action = 999999

    response = client.post("/action/train_001/0/1/10", json={"action": large_action})
    assert response.status_code == 200

    get_response = client.get("/action/train_001/0/1/10")
    assert get_response.status_code == 200
    assert get_response.json()["action"] == large_action


def test_parallel_workers(client, mock_get_action_memory):
    """Test handling multiple parallel running instances."""
    num_envs = 5

    # Add actions for parallel workers
    for worker_id in range(num_envs):
        response = client.post(
            f"/action/train_001/{worker_id}/1/10", json={"action": worker_id * 10}
        )
        assert response.status_code == 200

    # Retrieve and verify each
    for worker_id in range(num_envs):
        response = client.get(f"/action/train_001/{worker_id}/1/10")
        assert response.status_code == 200
        assert response.json()["action"] == worker_id * 10


def test_episode_progression(client, mock_get_action_memory):
    """Test actions across multiple episodes."""
    for episode_id in range(3):
        for step in range(5):
            response = client.post(
                f"/action/train_001/0/{episode_id}/{step}",
                json={"action": episode_id * 100 + step},
            )
            assert response.status_code == 200

    # Verify specific actions
    response = client.get("/action/train_001/0/1/3")
    assert response.json()["action"] == 103


def test_train_id_with_special_characters(client, mock_get_action_memory):
    """Test train_id with special characters."""
    train_ids = ["train-001", "train_002", "train.003"]

    for train_id in train_ids:
        response = client.post(f"/action/{train_id}/0/1/10", json={"action": 1})
        assert response.status_code == 200

        get_response = client.get(f"/action/{train_id}/0/1/10")
        assert get_response.status_code == 200


def test_action_schema_with_executed_field(client, mock_get_action_memory):
    """Test action schema with executed field."""
    payload = {"action": 5, "executed": True}

    response = client.post("/action/train_001/0/1/10", json=payload)
    assert response.status_code == 200

    # Retrieve and verify executed field
    get_response = client.get("/action/train_001/0/1/10")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["action"] == 5
    assert data["executed"] is True


def test_action_schema_executed_defaults_to_false(client, mock_get_action_memory):
    """Test that executed field defaults to False when not provided."""
    payload = {"action": 3}

    response = client.post("/action/train_001/0/1/10", json=payload)
    assert response.status_code == 200

    # Retrieve and verify executed defaults to False
    get_response = client.get("/action/train_001/0/1/10")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["action"] == 3
    assert data["executed"] is False


def test_action_schema_executed_false_explicitly(client, mock_get_action_memory):
    """Test setting executed field explicitly to False."""
    payload = {"action": 7, "executed": False}

    response = client.post("/action/train_001/0/1/10", json=payload)
    assert response.status_code == 200

    get_response = client.get("/action/train_001/0/1/10")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["action"] == 7
    assert data["executed"] is False
