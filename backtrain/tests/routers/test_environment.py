"""Unit tests for the Environment State Communication Router."""

import pytest
from app.core.memory import Memory
from app.dependencies import get_environment_memory
from app.routers.environment import router
from app.schemas import EnvironmentSchema
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    """Create a FastAPI application with the environment router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def environment_memory():
    """Provide a fresh memory instance for each test."""
    return Memory(capacity=100)


@pytest.fixture
def mock_get_environment_memory(environment_memory, app):
    """Mock the get_environment_memory dependency."""

    def _get_memory():
        return environment_memory

    # Override the dependency in the app
    app.dependency_overrides[get_environment_memory] = _get_memory

    yield environment_memory

    # Clean up after test
    app.dependency_overrides.clear()


@pytest.fixture
def client(app, mock_get_environment_memory):
    """Create a test client for the FastAPI application."""
    return TestClient(app)


def test_check_environment_memory_health_empty(client, mock_get_environment_memory):
    """Test health check returns correct stats for empty memory."""
    response = client.get("/environment/health")

    assert response.status_code == 200
    data = response.json()
    assert data["size"] == 0
    assert data["capacity"] == 100
    assert data["remaining"] == 100
    assert data["usage_percent"] == 0.0


def test_check_environment_memory_health_with_data(client, mock_get_environment_memory):
    """Test health check returns correct stats when memory contains data."""
    memory = mock_get_environment_memory

    # Add some environment states
    for i in range(5):
        env_state = EnvironmentSchema(done=False, reward=float(i), data={"step": i})
        memory.add(("train_001", 0, 1, i), env_state)

    response = client.get("/environment/health")

    assert response.status_code == 200
    data = response.json()
    assert data["size"] == 5
    assert data["capacity"] == 100
    assert data["remaining"] == 95
    assert data["usage_percent"] == 5.0


def test_add_environment_step_success(client, mock_get_environment_memory):
    """Test successfully adding an environment state."""
    payload = {
        "done": False,
        "reward": 1.5,
        "data": {"position": [0.1, 0.2], "velocity": 0.5},
    }

    response = client.post("/environment/train_001/0/1/10", json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "success"}

    # Verify it was stored
    memory = mock_get_environment_memory
    key = ("train_001", 0, 1, 10)
    assert key in memory


def test_add_environment_step_multiple_states(client, mock_get_environment_memory):
    """Test adding multiple environment states with different keys."""
    states = [
        ("train_001", 0, 1, 0, False, 0.0),
        ("train_001", 0, 1, 1, False, 1.0),
        ("train_001", 0, 2, 0, False, 0.5),
        ("train_001", 1, 1, 0, True, 10.0),
    ]

    for train_id, worker_id, episode_id, step, done, reward in states:
        response = client.post(
            f"/environment/{train_id}/{worker_id}/{episode_id}/{step}",
            json={"done": done, "reward": reward, "data": {}},
        )
        assert response.status_code == 200

    memory = mock_get_environment_memory
    assert len(memory) == 4


def test_add_environment_step_invalid_done_type(client, mock_get_environment_memory):
    """Test validation error when done field is not a boolean."""
    payload = {"done": "not_a_boolean", "reward": 1.0, "data": {}}

    response = client.post("/environment/train_001/0/1/10", json=payload)

    assert response.status_code == 422


def test_add_environment_step_invalid_reward_type(client, mock_get_environment_memory):
    """Test validation error when reward is not a number."""
    payload = {"done": False, "reward": "not_a_number", "data": {}}

    response = client.post("/environment/train_001/0/1/10", json=payload)

    assert response.status_code == 422


def test_add_environment_step_missing_done_field(client, mock_get_environment_memory):
    """Test validation error when done field is missing."""
    payload = {"reward": 1.0, "data": {}}

    response = client.post("/environment/train_001/0/1/10", json=payload)

    assert response.status_code == 422


def test_add_environment_step_missing_reward_field(client, mock_get_environment_memory):
    """Test validation error when reward field is missing."""
    payload = {"done": False, "data": {}}

    response = client.post("/environment/train_001/0/1/10", json=payload)

    assert response.status_code == 422


def test_add_environment_step_missing_data_field(client, mock_get_environment_memory):
    """Test validation error when data field is missing."""
    payload = {"done": False, "reward": 1.0}

    response = client.post("/environment/train_001/0/1/10", json=payload)

    assert response.status_code == 422


def test_add_environment_step_invalid_train_id(client, mock_get_environment_memory):
    """Test validation error for empty train_id."""
    payload = {"done": False, "reward": 1.0, "data": {}}

    response = client.post("/environment//0/1/10", json=payload)

    assert response.status_code == 404


def test_add_environment_step_negative_worker_id(client, mock_get_environment_memory):
    """Test validation error for negative worker_id."""
    payload = {"done": False, "reward": 1.0, "data": {}}

    response = client.post("/environment/train_001/-1/1/10", json=payload)

    assert response.status_code == 422


def test_add_environment_step_negative_episode_id(client, mock_get_environment_memory):
    """Test validation error for negative episode_id."""
    payload = {"done": False, "reward": 1.0, "data": {}}

    response = client.post("/environment/train_001/0/-1/10", json=payload)

    assert response.status_code == 422


def test_add_environment_step_negative_step(client, mock_get_environment_memory):
    """Test validation error for negative step."""
    payload = {"done": False, "reward": 1.0, "data": {}}

    response = client.post("/environment/train_001/0/1/-1", json=payload)

    assert response.status_code == 422


def test_get_environment_step_success(client, mock_get_environment_memory):
    """Test successfully retrieving an environment state."""
    memory = mock_get_environment_memory
    key = ("train_001", 0, 1, 10)
    env_state = EnvironmentSchema(
        done=False, reward=2.5, data={"position": [1.0, 2.0], "info": "test"}
    )
    memory.add(key, env_state)

    response = client.get("/environment/train_001/0/1/10")

    assert response.status_code == 200
    data = response.json()
    assert data["done"] is False
    assert data["reward"] == 2.5
    assert data["data"]["position"] == [1.0, 2.0]
    assert data["data"]["info"] == "test"


def test_get_environment_step_not_found(client, mock_get_environment_memory):
    """Test 404 error when environment state is not found."""
    response = client.get("/environment/train_001/0/1/10")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "Environment not found" in data["detail"]
    assert "train_id=train_001" in data["detail"]
    assert "worker_id=0" in data["detail"]
    assert "episode_id=1" in data["detail"]
    assert "step=10" in data["detail"]


def test_get_environment_step_multiple_retrievals(client, mock_get_environment_memory):
    """Test that get is non-destructive and state can be retrieved multiple times."""
    memory = mock_get_environment_memory
    key = ("train_001", 0, 1, 10)
    env_state = EnvironmentSchema(done=True, reward=10.0, data={"final": True})
    memory.add(key, env_state)

    # Retrieve multiple times
    response1 = client.get("/environment/train_001/0/1/10")
    response2 = client.get("/environment/train_001/0/1/10")
    response3 = client.get("/environment/train_001/0/1/10")

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response3.status_code == 200
    assert response1.json()["reward"] == 10.0
    assert response2.json()["reward"] == 10.0
    assert response3.json()["reward"] == 10.0


def test_get_environment_step_different_keys(client, mock_get_environment_memory):
    """Test retrieving environment states with different hierarchical keys."""
    memory = mock_get_environment_memory

    # Add states with different keys
    memory.add(
        ("train_001", 0, 1, 10), EnvironmentSchema(done=False, reward=1.0, data={})
    )
    memory.add(
        ("train_001", 1, 1, 10), EnvironmentSchema(done=False, reward=2.0, data={})
    )
    memory.add(
        ("train_001", 0, 2, 10), EnvironmentSchema(done=False, reward=3.0, data={})
    )
    memory.add(
        ("train_002", 0, 1, 10), EnvironmentSchema(done=False, reward=4.0, data={})
    )

    # Retrieve each one
    assert client.get("/environment/train_001/0/1/10").json()["reward"] == 1.0
    assert client.get("/environment/train_001/1/1/10").json()["reward"] == 2.0
    assert client.get("/environment/train_001/0/2/10").json()["reward"] == 3.0
    assert client.get("/environment/train_002/0/1/10").json()["reward"] == 4.0


def test_add_and_get_environment_workflow(client, mock_get_environment_memory):
    """Test complete workflow of adding and retrieving an environment state."""
    # Add environment state
    add_response = client.post(
        "/environment/exp_001/2/5/100",
        json={
            "done": True,
            "reward": 42.5,
            "data": {"position": [10.0, 20.0], "velocity": [1.0, 2.0]},
        },
    )
    assert add_response.status_code == 200
    assert add_response.json()["status"] == "success"

    # Retrieve environment state
    get_response = client.get("/environment/exp_001/2/5/100")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["done"] is True
    assert data["reward"] == 42.5
    assert data["data"]["position"] == [10.0, 20.0]


def test_environment_with_zero_reward(client, mock_get_environment_memory):
    """Test handling environment states with zero reward."""
    response = client.post(
        "/environment/train_001/0/0/0", json={"done": False, "reward": 0.0, "data": {}}
    )
    assert response.status_code == 200

    get_response = client.get("/environment/train_001/0/0/0")
    assert get_response.status_code == 200
    assert get_response.json()["reward"] == 0.0


def test_environment_with_negative_reward(client, mock_get_environment_memory):
    """Test handling environment states with negative reward."""
    response = client.post(
        "/environment/train_001/0/1/10",
        json={"done": False, "reward": -10.5, "data": {}},
    )
    assert response.status_code == 200

    get_response = client.get("/environment/train_001/0/1/10")
    assert get_response.status_code == 200
    assert get_response.json()["reward"] == -10.5


def test_environment_with_complex_data_structure(client, mock_get_environment_memory):
    """Test handling environment states with complex nested data."""
    complex_data = {
        "sensors": [0.1, 0.2, 0.3],
        "camera": {"width": 640, "height": 480, "pixels": [[1, 2], [3, 4]]},
        "metadata": {"timestamp": 123456789, "agent_id": "agent_001"},
    }

    response = client.post(
        "/environment/train_001/0/1/10",
        json={"done": False, "reward": 1.0, "data": complex_data},
    )
    assert response.status_code == 200

    get_response = client.get("/environment/train_001/0/1/10")
    assert get_response.status_code == 200
    retrieved_data = get_response.json()["data"]
    assert retrieved_data["sensors"] == [0.1, 0.2, 0.3]
    assert retrieved_data["camera"]["width"] == 640
    assert retrieved_data["metadata"]["agent_id"] == "agent_001"


def test_environment_terminal_state(client, mock_get_environment_memory):
    """Test handling terminal environment states."""
    response = client.post(
        "/environment/train_001/0/1/100",
        json={
            "done": True,
            "reward": 100.0,
            "data": {"reason": "goal_reached", "final_score": 100},
        },
    )
    assert response.status_code == 200

    get_response = client.get("/environment/train_001/0/1/100")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["done"] is True
    assert data["data"]["reason"] == "goal_reached"


def test_parallel_workers(client, mock_get_environment_memory):
    """Test handling multiple parallel running instances."""
    num_envs = 5

    # Add environment states for parallel workers
    for worker_id in range(num_envs):
        response = client.post(
            f"/environment/train_001/{worker_id}/1/10",
            json={
                "done": False,
                "reward": float(worker_id * 10),
                "data": {"worker_id": worker_id},
            },
        )
        assert response.status_code == 200

    # Retrieve and verify each
    for worker_id in range(num_envs):
        response = client.get(f"/environment/train_001/{worker_id}/1/10")
        assert response.status_code == 200
        data = response.json()
        assert data["reward"] == worker_id * 10
        assert data["data"]["worker_id"] == worker_id


def test_episode_progression(client, mock_get_environment_memory):
    """Test environment states across multiple episodes."""
    for episode_id in range(3):
        for step in range(5):
            response = client.post(
                f"/environment/train_001/0/{episode_id}/{step}",
                json={
                    "done": (step == 4),
                    "reward": float(episode_id * 100 + step),
                    "data": {"episode": episode_id, "step": step},
                },
            )
            assert response.status_code == 200

    # Verify specific states
    response = client.get("/environment/train_001/0/1/3")
    data = response.json()
    assert data["reward"] == 103.0
    assert data["done"] is False

    # Verify terminal state
    response = client.get("/environment/train_001/0/1/4")
    data = response.json()
    assert data["done"] is True


def test_train_id_with_special_characters(client, mock_get_environment_memory):
    """Test train_id with special characters."""
    train_ids = ["train-001", "train_002", "train.003"]

    for train_id in train_ids:
        response = client.post(
            f"/environment/{train_id}/0/1/10",
            json={"done": False, "reward": 1.0, "data": {}},
        )
        assert response.status_code == 200

        get_response = client.get(f"/environment/{train_id}/0/1/10")
        assert get_response.status_code == 200


def test_environment_with_empty_data(client, mock_get_environment_memory):
    """Test handling environment states with empty data dictionary."""
    response = client.post(
        "/environment/train_001/0/1/10", json={"done": False, "reward": 1.0, "data": {}}
    )
    assert response.status_code == 200

    get_response = client.get("/environment/train_001/0/1/10")
    assert get_response.status_code == 200
    assert get_response.json()["data"] == {}


def test_environment_with_large_reward(client, mock_get_environment_memory):
    """Test handling environment states with large reward values."""
    large_reward = 1e10

    response = client.post(
        "/environment/train_001/0/1/10",
        json={"done": False, "reward": large_reward, "data": {}},
    )
    assert response.status_code == 200

    get_response = client.get("/environment/train_001/0/1/10")
    assert get_response.status_code == 200
    assert get_response.json()["reward"] == large_reward
