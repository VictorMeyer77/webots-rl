"""Unit tests for the Observation Communication Router."""

import pytest
from app.core.memory import Memory
from app.dependencies import get_observation_memory
from app.routers.observation import router
from app.schemas import ObservationSchema
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    """Create a FastAPI application with the observation router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def observation_memory():
    """Provide a fresh memory instance for each test."""
    return Memory(capacity=100)


@pytest.fixture
def mock_get_observation_memory(observation_memory, app):
    """Mock the get_observation_memory dependency."""

    def _get_memory():
        return observation_memory

    # Override the dependency in the app
    app.dependency_overrides[get_observation_memory] = _get_memory

    yield observation_memory

    # Clean up after test
    app.dependency_overrides.clear()


@pytest.fixture
def client(app, mock_get_observation_memory):
    """Create a test client for the FastAPI application."""
    return TestClient(app)


def test_add_observation_step_success(client, mock_get_observation_memory):
    """Test successfully adding an observation."""
    payload = {"data": {"position": [1.0, 2.0], "velocity": [0.5, -0.3]}}

    response = client.post("/observation/train_001/0/1/10", json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "stored"}

    # Verify it was stored
    memory = mock_get_observation_memory
    key = ("train_001", 0, 1, 10)
    assert key in memory


def test_add_observation_step_multiple_observations(
    client, mock_get_observation_memory
):
    """Test adding multiple observations with different keys."""
    observations = [
        ("train_001", 0, 1, 0, {"sensors": [0.1, 0.2]}),
        ("train_001", 0, 1, 1, {"sensors": [0.3, 0.4]}),
        ("train_001", 0, 2, 0, {"sensors": [0.5, 0.6]}),
        ("train_001", 1, 1, 0, {"sensors": [0.7, 0.8]}),
    ]

    for train_id, env_id, episode_id, step, data in observations:
        response = client.post(
            f"/observation/{train_id}/{env_id}/{episode_id}/{step}", json={"data": data}
        )
        assert response.status_code == 200

    memory = mock_get_observation_memory
    assert len(memory) == 4


def test_add_observation_step_missing_data_field(client, mock_get_observation_memory):
    """Test validation error when data field is missing."""
    payload = {}

    response = client.post("/observation/train_001/0/1/10", json=payload)

    assert response.status_code == 422


def test_add_observation_step_invalid_data_type(client, mock_get_observation_memory):
    """Test validation error when data is not a dictionary."""
    payload = {"data": "not_a_dict"}

    response = client.post("/observation/train_001/0/1/10", json=payload)

    assert response.status_code == 422


def test_add_observation_step_invalid_train_id(client, mock_get_observation_memory):
    """Test validation error for empty train_id."""
    payload = {"data": {"value": 1.0}}

    response = client.post("/observation//0/1/10", json=payload)

    assert response.status_code == 404


def test_add_observation_step_negative_env_id(client, mock_get_observation_memory):
    """Test validation error for negative env_id."""
    payload = {"data": {"value": 1.0}}

    response = client.post("/observation/train_001/-1/1/10", json=payload)

    assert response.status_code == 422


def test_add_observation_step_negative_episode_id(client, mock_get_observation_memory):
    """Test validation error for negative episode_id."""
    payload = {"data": {"value": 1.0}}

    response = client.post("/observation/train_001/0/-1/10", json=payload)

    assert response.status_code == 422


def test_add_observation_step_negative_step(client, mock_get_observation_memory):
    """Test validation error for negative step."""
    payload = {"data": {"value": 1.0}}

    response = client.post("/observation/train_001/0/1/-1", json=payload)

    assert response.status_code == 422


def test_get_observation_step_success(client, mock_get_observation_memory):
    """Test successfully retrieving an observation."""
    memory = mock_get_observation_memory
    key = ("train_001", 0, 1, 10)
    observation = ObservationSchema(
        data={"position": [1.0, 2.0], "velocity": [0.5, -0.3]}
    )
    memory.add(key, observation)

    response = client.get("/observation/train_001/0/1/10")

    assert response.status_code == 200
    data = response.json()
    assert data["data"]["position"] == [1.0, 2.0]
    assert data["data"]["velocity"] == [0.5, -0.3]


def test_get_observation_step_not_found(client, mock_get_observation_memory):
    """Test 404 error when observation is not found."""
    response = client.get("/observation/train_001/0/1/10")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "Observation not found" in data["detail"]
    assert "train_id=train_001" in data["detail"]
    assert "env_id=0" in data["detail"]
    assert "episode_id=1" in data["detail"]
    assert "step=10" in data["detail"]


def test_get_observation_step_multiple_retrievals(client, mock_get_observation_memory):
    """Test that get is non-destructive and observation can be retrieved multiple times."""
    memory = mock_get_observation_memory
    key = ("train_001", 0, 1, 10)
    observation = ObservationSchema(data={"value": 42.0})
    memory.add(key, observation)

    # Retrieve multiple times
    response1 = client.get("/observation/train_001/0/1/10")
    response2 = client.get("/observation/train_001/0/1/10")
    response3 = client.get("/observation/train_001/0/1/10")

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response3.status_code == 200
    assert response1.json()["data"]["value"] == 42.0
    assert response2.json()["data"]["value"] == 42.0
    assert response3.json()["data"]["value"] == 42.0


def test_get_observation_step_different_keys(client, mock_get_observation_memory):
    """Test retrieving observations with different hierarchical keys."""
    memory = mock_get_observation_memory

    # Add observations with different keys
    memory.add(("train_001", 0, 1, 10), ObservationSchema(data={"id": 1}))
    memory.add(("train_001", 1, 1, 10), ObservationSchema(data={"id": 2}))
    memory.add(("train_001", 0, 2, 10), ObservationSchema(data={"id": 3}))
    memory.add(("train_002", 0, 1, 10), ObservationSchema(data={"id": 4}))

    # Retrieve each one
    assert client.get("/observation/train_001/0/1/10").json()["data"]["id"] == 1
    assert client.get("/observation/train_001/1/1/10").json()["data"]["id"] == 2
    assert client.get("/observation/train_001/0/2/10").json()["data"]["id"] == 3
    assert client.get("/observation/train_002/0/1/10").json()["data"]["id"] == 4


def test_add_and_get_observation_workflow(client, mock_get_observation_memory):
    """Test complete workflow of adding and retrieving an observation."""
    # Add observation
    add_response = client.post(
        "/observation/exp_001/2/5/100",
        json={"data": {"position": [10.0, 20.0], "velocity": [1.0, 2.0]}},
    )
    assert add_response.status_code == 200
    assert add_response.json()["status"] == "stored"

    # Retrieve observation
    get_response = client.get("/observation/exp_001/2/5/100")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["data"]["position"] == [10.0, 20.0]
    assert data["data"]["velocity"] == [1.0, 2.0]


def test_observation_with_empty_data(client, mock_get_observation_memory):
    """Test handling observations with empty data dictionary."""
    response = client.post("/observation/train_001/0/1/10", json={"data": {}})
    assert response.status_code == 200

    get_response = client.get("/observation/train_001/0/1/10")
    assert get_response.status_code == 200
    assert get_response.json()["data"] == {}


def test_observation_with_complex_nested_data(client, mock_get_observation_memory):
    """Test handling observations with complex nested data structures."""
    complex_data = {
        "sensors": [0.1, 0.2, 0.3, 0.4, 0.5],
        "camera": {"width": 640, "height": 480, "pixels": [[1, 2, 3], [4, 5, 6]]},
        "metadata": {
            "timestamp": 123456789,
            "agent_id": "agent_001",
            "flags": {"collision": False, "goal_reached": True},
        },
    }

    response = client.post("/observation/train_001/0/1/10", json={"data": complex_data})
    assert response.status_code == 200

    get_response = client.get("/observation/train_001/0/1/10")
    assert get_response.status_code == 200
    retrieved_data = get_response.json()["data"]
    assert retrieved_data["sensors"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert retrieved_data["camera"]["width"] == 640
    assert retrieved_data["metadata"]["agent_id"] == "agent_001"
    assert retrieved_data["metadata"]["flags"]["goal_reached"] is True


def test_observation_with_arrays(client, mock_get_observation_memory):
    """Test handling observations with array data."""
    array_data = {
        "image": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "lidar": [0.5, 1.2, 3.4, 2.1, 0.8],
    }

    response = client.post("/observation/train_001/0/1/10", json={"data": array_data})
    assert response.status_code == 200

    get_response = client.get("/observation/train_001/0/1/10")
    assert get_response.status_code == 200
    data = get_response.json()["data"]
    assert data["image"] == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert data["lidar"] == [0.5, 1.2, 3.4, 2.1, 0.8]


def test_observation_with_zero_values(client, mock_get_observation_memory):
    """Test handling observations with zero values."""
    response = client.post(
        "/observation/train_001/0/0/0", json={"data": {"value": 0.0, "count": 0}}
    )
    assert response.status_code == 200

    get_response = client.get("/observation/train_001/0/0/0")
    assert get_response.status_code == 200
    data = get_response.json()["data"]
    assert data["value"] == 0.0
    assert data["count"] == 0


def test_observation_with_negative_values(client, mock_get_observation_memory):
    """Test handling observations with negative values."""
    response = client.post(
        "/observation/train_001/0/1/10",
        json={"data": {"position": [-1.5, -2.3], "velocity": [-0.5, -0.8]}},
    )
    assert response.status_code == 200

    get_response = client.get("/observation/train_001/0/1/10")
    assert get_response.status_code == 200
    data = get_response.json()["data"]
    assert data["position"] == [-1.5, -2.3]
    assert data["velocity"] == [-0.5, -0.8]


def test_parallel_environments(client, mock_get_observation_memory):
    """Test handling multiple parallel environment instances."""
    num_envs = 5

    # Add observations for parallel environments
    for env_id in range(num_envs):
        response = client.post(
            f"/observation/train_001/{env_id}/1/10",
            json={"data": {"env_id": env_id, "value": float(env_id * 10)}},
        )
        assert response.status_code == 200

    # Retrieve and verify each
    for env_id in range(num_envs):
        response = client.get(f"/observation/train_001/{env_id}/1/10")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["env_id"] == env_id
        assert data["value"] == env_id * 10


def test_episode_progression(client, mock_get_observation_memory):
    """Test observations across multiple episodes."""
    for episode_id in range(3):
        for step in range(5):
            response = client.post(
                f"/observation/train_001/0/{episode_id}/{step}",
                json={
                    "data": {
                        "episode": episode_id,
                        "step": step,
                        "value": episode_id * 100 + step,
                    }
                },
            )
            assert response.status_code == 200

    # Verify specific observations
    response = client.get("/observation/train_001/0/1/3")
    data = response.json()["data"]
    assert data["episode"] == 1
    assert data["step"] == 3
    assert data["value"] == 103


def test_train_id_with_special_characters(client, mock_get_observation_memory):
    """Test train_id with special characters."""
    train_ids = ["train-001", "train_002", "train.003"]

    for train_id in train_ids:
        response = client.post(
            f"/observation/{train_id}/0/1/10", json={"data": {"train_id": train_id}}
        )
        assert response.status_code == 200

        get_response = client.get(f"/observation/{train_id}/0/1/10")
        assert get_response.status_code == 200
        assert get_response.json()["data"]["train_id"] == train_id


def test_observation_with_boolean_values(client, mock_get_observation_memory):
    """Test handling observations with boolean values."""
    response = client.post(
        "/observation/train_001/0/1/10",
        json={"data": {"collision": True, "goal_reached": False, "timeout": False}},
    )
    assert response.status_code == 200

    get_response = client.get("/observation/train_001/0/1/10")
    assert get_response.status_code == 200
    data = get_response.json()["data"]
    assert data["collision"] is True
    assert data["goal_reached"] is False
    assert data["timeout"] is False


def test_observation_with_string_values(client, mock_get_observation_memory):
    """Test handling observations with string values."""
    response = client.post(
        "/observation/train_001/0/1/10",
        json={"data": {"state": "running", "mode": "exploration", "agent_type": "DQN"}},
    )
    assert response.status_code == 200

    get_response = client.get("/observation/train_001/0/1/10")
    assert get_response.status_code == 200
    data = get_response.json()["data"]
    assert data["state"] == "running"
    assert data["mode"] == "exploration"
    assert data["agent_type"] == "DQN"


def test_observation_with_mixed_types(client, mock_get_observation_memory):
    """Test handling observations with mixed data types."""
    mixed_data = {
        "position": [1.0, 2.0],
        "step": 42,
        "done": False,
        "info": "test",
        "metadata": {"timestamp": 123456, "valid": True},
    }

    response = client.post("/observation/train_001/0/1/10", json={"data": mixed_data})
    assert response.status_code == 200

    get_response = client.get("/observation/train_001/0/1/10")
    assert get_response.status_code == 200
    data = get_response.json()["data"]
    assert data["position"] == [1.0, 2.0]
    assert data["step"] == 42
    assert data["done"] is False
    assert data["info"] == "test"
    assert data["metadata"]["timestamp"] == 123456


def test_observation_with_large_data(client, mock_get_observation_memory):
    """Test handling observations with large data structures."""
    large_array = [i * 0.1 for i in range(1000)]

    response = client.post(
        "/observation/train_001/0/1/10", json={"data": {"sensors": large_array}}
    )
    assert response.status_code == 200

    get_response = client.get("/observation/train_001/0/1/10")
    assert get_response.status_code == 200
    data = get_response.json()["data"]
    assert len(data["sensors"]) == 1000
    assert data["sensors"][0] == 0.0
    assert data["sensors"][999] == 99.9
