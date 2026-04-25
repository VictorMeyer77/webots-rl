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


def test_get_environment_batch_all_found(client, mock_get_environment_memory):
    """Test batch retrieval when all requested environment states are found."""
    memory = mock_get_environment_memory

    # Add environment states
    memory.add(
        ("train_001", 0, 1, 10), EnvironmentSchema(done=False, reward=1.0, data={})
    )
    memory.add(
        ("train_001", 0, 1, 11), EnvironmentSchema(done=False, reward=2.0, data={})
    )
    memory.add(
        ("train_001", 0, 1, 12), EnvironmentSchema(done=True, reward=3.0, data={})
    )

    # Batch request
    payload = {
        "keys": [
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 10},
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 11},
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 12},
        ]
    }

    response = client.post("/environment/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert data["missing"] == 0
    assert len(data["results"]) == 3

    # Verify each result
    assert data["results"][0]["value"]["reward"] == 1.0
    assert data["results"][0]["value"]["done"] is False
    assert data["results"][1]["value"]["reward"] == 2.0
    assert data["results"][2]["value"]["reward"] == 3.0
    assert data["results"][2]["value"]["done"] is True


def test_get_environment_batch_all_missing(client, mock_get_environment_memory):
    """Test batch retrieval when none of the requested environment states are found."""
    payload = {
        "keys": [
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 10},
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 11},
        ]
    }

    response = client.post("/environment/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["missing"] == 2
    assert len(data["results"]) == 2

    # Verify all results are not found
    assert data["results"][0]["value"] is None
    assert data["results"][1]["value"] is None


def test_get_environment_batch_mixed_results(client, mock_get_environment_memory):
    """Test batch retrieval with a mix of found and missing environment states."""
    memory = mock_get_environment_memory

    # Add only some environment states
    memory.add(
        ("train_001", 0, 1, 10), EnvironmentSchema(done=False, reward=1.0, data={})
    )
    memory.add(
        ("train_001", 0, 1, 12), EnvironmentSchema(done=True, reward=3.0, data={})
    )

    payload = {
        "keys": [
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 10},
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 11},
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 12},
        ]
    }

    response = client.post("/environment/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert data["missing"] == 1

    # Verify specific results
    assert data["results"][0]["value"]["reward"] == 1.0
    assert data["results"][1]["value"] is None
    assert data["results"][2]["value"]["reward"] == 3.0


def test_get_environment_batch_empty_keys(client, mock_get_environment_memory):
    """Test batch retrieval with an empty list of keys."""
    payload = {"keys": []}

    response = client.post("/environment/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["found"] == 0
    assert data["missing"] == 0
    assert len(data["results"]) == 0


def test_get_environment_batch_single_key(client, mock_get_environment_memory):
    """Test batch retrieval with a single key."""
    memory = mock_get_environment_memory
    memory.add(
        ("train_001", 0, 1, 10),
        EnvironmentSchema(done=False, reward=5.5, data={"position": [10.0, 20.0]}),
    )

    payload = {
        "keys": [{"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 10}]
    }

    response = client.post("/environment/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["found"] == 1
    assert data["missing"] == 0
    assert data["results"][0]["value"]["reward"] == 5.5
    assert data["results"][0]["value"]["data"]["position"] == [10.0, 20.0]


def test_get_environment_batch_different_keys(client, mock_get_environment_memory):
    """Test batch retrieval with environment states from different hierarchical levels."""
    memory = mock_get_environment_memory

    # Add environment states with different train_id, worker_id, and episode_id
    memory.add(
        ("train_001", 0, 1, 10), EnvironmentSchema(done=False, reward=1.0, data={})
    )
    memory.add(
        ("train_001", 1, 1, 10), EnvironmentSchema(done=False, reward=2.0, data={})
    )
    memory.add(
        ("train_002", 0, 1, 10), EnvironmentSchema(done=False, reward=3.0, data={})
    )
    memory.add(
        ("train_001", 0, 2, 10), EnvironmentSchema(done=True, reward=4.0, data={})
    )

    payload = {
        "keys": [
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 10},
            {"train_id": "train_001", "worker_id": 1, "episode_id": 1, "step": 10},
            {"train_id": "train_002", "worker_id": 0, "episode_id": 1, "step": 10},
            {"train_id": "train_001", "worker_id": 0, "episode_id": 2, "step": 10},
        ]
    }

    response = client.post("/environment/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 4
    assert data["found"] == 4
    assert data["missing"] == 0

    # Verify each environment state matches
    assert data["results"][0]["value"]["reward"] == 1.0
    assert data["results"][0]["value"]["done"] is False
    assert data["results"][1]["value"]["reward"] == 2.0
    assert data["results"][2]["value"]["reward"] == 3.0
    assert data["results"][3]["value"]["reward"] == 4.0
    assert data["results"][3]["value"]["done"] is True


def test_get_environment_batch_invalid_payload(client, mock_get_environment_memory):
    """Test validation error for invalid batch request payload."""
    # Missing keys field
    payload = {}

    response = client.post("/environment/batch", json=payload)

    assert response.status_code == 422


def test_get_environment_batch_invalid_key_structure(
    client, mock_get_environment_memory
):
    """Test validation error for invalid key structure in batch request."""
    # Missing required fields in key
    payload = {"keys": [{"train_id": "train_001", "worker_id": 0}]}

    response = client.post("/environment/batch", json=payload)

    assert response.status_code == 422


def test_get_environment_batch_negative_values(client, mock_get_environment_memory):
    """Test validation error for negative values in batch request keys."""
    payload = {
        "keys": [
            {"train_id": "train_001", "worker_id": -1, "episode_id": 1, "step": 10}
        ]
    }

    response = client.post("/environment/batch", json=payload)

    assert response.status_code == 422


def test_get_environment_batch_complex_data(client, mock_get_environment_memory):
    """Test batch retrieval with complex nested environment data."""
    memory = mock_get_environment_memory

    # Add environment states with complex data
    memory.add(
        ("train_001", 0, 1, 10),
        EnvironmentSchema(
            done=False,
            reward=1.5,
            data={
                "position": [0.1, 0.2, 0.3],
                "velocity": [0.5, 0.6],
                "metadata": {"timestamp": 100, "valid": True},
            },
        ),
    )
    memory.add(
        ("train_001", 0, 1, 11),
        EnvironmentSchema(
            done=True,
            reward=10.0,
            data={
                "final_state": {"score": 100, "reason": "goal_reached"},
                "stats": {"steps": 50, "time": 123.45},
            },
        ),
    )

    payload = {
        "keys": [
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 10},
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 11},
        ]
    }

    response = client.post("/environment/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["found"] == 2
    assert data["missing"] == 0

    # Verify complex data structure is preserved
    assert data["results"][0]["value"]["data"]["position"] == [0.1, 0.2, 0.3]
    assert data["results"][0]["value"]["data"]["metadata"]["valid"] is True
    assert data["results"][1]["value"]["done"] is True
    assert data["results"][1]["value"]["data"]["final_state"]["score"] == 100
    assert data["results"][1]["value"]["data"]["stats"]["steps"] == 50


def test_get_environment_batch_with_negative_rewards(
    client, mock_get_environment_memory
):
    """Test batch retrieval with negative reward values."""
    memory = mock_get_environment_memory

    memory.add(
        ("train_001", 0, 1, 10), EnvironmentSchema(done=False, reward=-1.5, data={})
    )
    memory.add(
        ("train_001", 0, 1, 11), EnvironmentSchema(done=False, reward=-10.0, data={})
    )

    payload = {
        "keys": [
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 10},
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 11},
        ]
    }

    response = client.post("/environment/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["found"] == 2
    assert data["results"][0]["value"]["reward"] == -1.5
    assert data["results"][1]["value"]["reward"] == -10.0


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


# Tests for post_environment_batch endpoint


def test_post_environment_batch_single_item(client, mock_get_environment_memory):
    """Test publishing a single environment state in a batch request."""
    payload = {
        "items": [
            {
                "key": {
                    "train_id": "exp_001",
                    "worker_id": 0,
                    "episode_id": 5,
                    "step": 10,
                },
                "value": {
                    "done": False,
                    "reward": 1.5,
                    "data": {"position": [0.5, 1.2]},
                },
            }
        ]
    }

    response = client.post("/environment/batch/publish", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total"] == 1

    # Verify the data was actually stored in memory
    memory = mock_get_environment_memory
    key = ("exp_001", 0, 5, 10)
    stored_value = memory.get(key)
    assert stored_value is not None
    assert stored_value.reward == 1.5
    assert stored_value.done is False
    assert stored_value.data["position"] == [0.5, 1.2]


def test_post_environment_batch_multiple_items(client, mock_get_environment_memory):
    """Test publishing multiple environment states in a single batch request."""
    payload = {
        "items": [
            {
                "key": {
                    "train_id": "exp_001",
                    "worker_id": 0,
                    "episode_id": 5,
                    "step": 10,
                },
                "value": {
                    "done": False,
                    "reward": 1.0,
                    "data": {"position": [0.1, 0.2]},
                },
            },
            {
                "key": {
                    "train_id": "exp_001",
                    "worker_id": 0,
                    "episode_id": 5,
                    "step": 11,
                },
                "value": {
                    "done": False,
                    "reward": 2.0,
                    "data": {"position": [0.2, 0.3]},
                },
            },
            {
                "key": {
                    "train_id": "exp_001",
                    "worker_id": 1,
                    "episode_id": 5,
                    "step": 10,
                },
                "value": {
                    "done": True,
                    "reward": 10.0,
                    "data": {"position": [1.0, 1.0]},
                },
            },
        ]
    }

    response = client.post("/environment/batch/publish", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total"] == 3

    # Verify all items were stored
    memory = mock_get_environment_memory
    key1 = ("exp_001", 0, 5, 10)
    key2 = ("exp_001", 0, 5, 11)
    key3 = ("exp_001", 1, 5, 10)

    assert memory.get(key1) is not None
    assert memory.get(key2) is not None
    assert memory.get(key3) is not None
    assert memory.get(key1).reward == 1.0
    assert memory.get(key2).reward == 2.0
    assert memory.get(key3).reward == 10.0
    assert memory.get(key3).done is True


def test_post_environment_batch_empty(client, mock_get_environment_memory):
    """Test publishing an empty batch request."""
    payload = {"items": []}

    response = client.post("/environment/batch/publish", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total"] == 0


def test_post_environment_batch_overwrites_existing_keys(
    client, mock_get_environment_memory
):
    """Test that publishing with existing keys overwrites the previous values."""
    key = {"train_id": "exp_001", "worker_id": 0, "episode_id": 5, "step": 10}

    # First publish
    payload1 = {
        "items": [
            {
                "key": key,
                "value": {
                    "done": False,
                    "reward": 1.0,
                    "data": {"position": [0.1, 0.2]},
                },
            }
        ]
    }
    response1 = client.post("/environment/batch/publish", json=payload1)
    assert response1.status_code == 200

    # Second publish with same key but different value
    payload2 = {
        "items": [
            {
                "key": key,
                "value": {
                    "done": True,
                    "reward": 5.0,
                    "data": {"position": [0.5, 0.6]},
                },
            }
        ]
    }
    response2 = client.post("/environment/batch/publish", json=payload2)
    assert response2.status_code == 200

    # Verify the value was overwritten
    memory = mock_get_environment_memory
    tuple_key = (key["train_id"], key["worker_id"], key["episode_id"], key["step"])
    stored_value = memory.get(tuple_key)
    assert stored_value.reward == 5.0
    assert stored_value.done is True


def test_post_environment_batch_different_train_ids(
    client, mock_get_environment_memory
):
    """Test publishing environment states for different training sessions."""
    payload = {
        "items": [
            {
                "key": {
                    "train_id": "exp_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 0,
                },
                "value": {"done": False, "reward": 1.0, "data": {}},
            },
            {
                "key": {
                    "train_id": "exp_002",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 0,
                },
                "value": {"done": False, "reward": 2.0, "data": {}},
            },
        ]
    }

    response = client.post("/environment/batch/publish", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2

    # Verify both were stored with different train_ids
    memory = mock_get_environment_memory
    key1 = ("exp_001", 0, 1, 0)
    key2 = ("exp_002", 0, 1, 0)
    assert memory.get(key1).reward == 1.0
    assert memory.get(key2).reward == 2.0


def test_post_environment_batch_negative_rewards(client, mock_get_environment_memory):
    """Test publishing environment states with negative reward values."""
    payload = {
        "items": [
            {
                "key": {
                    "train_id": "exp_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 0,
                },
                "value": {
                    "done": False,
                    "reward": -10.5,
                    "data": {"error": "collision"},
                },
            }
        ]
    }

    response = client.post("/environment/batch/publish", json=payload)

    assert response.status_code == 200
    memory = mock_get_environment_memory
    key = ("exp_001", 0, 1, 0)
    stored_value = memory.get(key)
    assert stored_value.reward == -10.5


def test_post_environment_batch_complex_data(client, mock_get_environment_memory):
    """Test publishing environment states with complex nested data structures."""
    complex_data = {
        "sensors": {
            "lidar": [0.1, 0.2, 0.3, 0.4, 0.5],
            "camera": {"width": 640, "height": 480, "fps": 30},
        },
        "position": [1.5, 2.3, 0.0],
        "rotation": [0.0, 0.0, 45.0],
        "metadata": {"timestamp": 1234567890, "frame_id": 42},
    }

    payload = {
        "items": [
            {
                "key": {
                    "train_id": "exp_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 0,
                },
                "value": {"done": False, "reward": 1.0, "data": complex_data},
            }
        ]
    }

    response = client.post("/environment/batch/publish", json=payload)

    assert response.status_code == 200
    memory = mock_get_environment_memory
    key = ("exp_001", 0, 1, 0)
    stored_value = memory.get(key)
    assert stored_value.data == complex_data


def test_post_environment_batch_validation_error_negative_worker_id(
    client, mock_get_environment_memory
):
    """Test validation error when worker_id is negative."""
    # Negative worker_id (should be >= 0)
    payload = {
        "items": [
            {
                "key": {
                    "train_id": "exp_001",
                    "worker_id": -1,
                    "episode_id": 1,
                    "step": 0,
                },
                "value": {"done": False, "reward": 1.0, "data": {}},
            }
        ]
    }

    response = client.post("/environment/batch/publish", json=payload)
    assert response.status_code == 422


def test_post_environment_batch_validation_error_empty_train_id(
    client, mock_get_environment_memory
):
    """Test validation error when train_id is empty."""
    # Empty train_id (should have min_length=1)
    payload = {
        "items": [
            {
                "key": {"train_id": "", "worker_id": 0, "episode_id": 1, "step": 0},
                "value": {"done": False, "reward": 1.0, "data": {}},
            }
        ]
    }

    response = client.post("/environment/batch/publish", json=payload)
    assert response.status_code == 422


def test_post_environment_batch_large_batch(client, mock_get_environment_memory):
    """Test publishing a large batch of environment states."""
    # Create 50 items
    items = [
        {
            "key": {"train_id": "exp_001", "worker_id": 0, "episode_id": 1, "step": i},
            "value": {"done": i == 49, "reward": float(i), "data": {"step": i}},
        }
        for i in range(50)
    ]

    payload = {"items": items}

    response = client.post("/environment/batch/publish", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total"] == 50

    # Verify a few random items were stored
    memory = mock_get_environment_memory
    assert memory.get(("exp_001", 0, 1, 0)) is not None
    assert memory.get(("exp_001", 0, 1, 25)) is not None
    assert memory.get(("exp_001", 0, 1, 49)) is not None
    assert memory.get(("exp_001", 0, 1, 49)).done is True


def test_post_environment_batch_zero_reward(client, mock_get_environment_memory):
    """Test publishing environment state with zero reward."""
    payload = {
        "items": [
            {
                "key": {
                    "train_id": "exp_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 0,
                },
                "value": {"done": False, "reward": 0.0, "data": {}},
            }
        ]
    }

    response = client.post("/environment/batch/publish", json=payload)

    assert response.status_code == 200
    memory = mock_get_environment_memory
    key = ("exp_001", 0, 1, 0)
    stored_value = memory.get(key)
    assert stored_value.reward == 0.0


def test_post_environment_batch_terminal_state(client, mock_get_environment_memory):
    """Test publishing a terminal environment state (done=True)."""
    payload = {
        "items": [
            {
                "key": {
                    "train_id": "exp_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 100,
                },
                "value": {
                    "done": True,
                    "reward": 100.0,
                    "data": {"reason": "goal_reached"},
                },
            }
        ]
    }

    response = client.post("/environment/batch/publish", json=payload)

    assert response.status_code == 200
    memory = mock_get_environment_memory
    key = ("exp_001", 0, 1, 100)
    stored_value = memory.get(key)
    assert stored_value.done is True
    assert stored_value.reward == 100.0


def test_post_environment_batch_parallel_workers(client, mock_get_environment_memory):
    """Test publishing environment states from multiple parallel workers."""
    payload = {
        "items": [
            {
                "key": {
                    "train_id": "exp_001",
                    "worker_id": i,
                    "episode_id": 1,
                    "step": 0,
                },
                "value": {
                    "done": False,
                    "reward": float(i * 10),
                    "data": {"worker_id": i},
                },
            }
            for i in range(5)
        ]
    }

    response = client.post("/environment/batch/publish", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 5

    # Verify all workers' states were stored
    memory = mock_get_environment_memory
    for i in range(5):
        key = ("exp_001", i, 1, 0)
        stored_value = memory.get(key)
        assert stored_value is not None
        assert stored_value.reward == float(i * 10)


def test_post_environment_batch_response_schema(client, mock_get_environment_memory):
    """Test that the response schema matches BatchPostResponseSchema."""
    payload = {
        "items": [
            {
                "key": {
                    "train_id": "exp_001",
                    "worker_id": 0,
                    "episode_id": 5,
                    "step": 10,
                },
                "value": {
                    "done": False,
                    "reward": 1.5,
                    "data": {"position": [0.5, 1.2]},
                },
            }
        ]
    }

    response = client.post("/environment/batch/publish", json=payload)

    assert response.status_code == 200
    data = response.json()

    # Verify response has expected fields
    assert "status" in data
    assert "total" in data
    assert isinstance(data["status"], str)
    assert isinstance(data["total"], int)
    assert data["status"] == "success"


def test_post_environment_batch_mixed_terminal_states(
    client, mock_get_environment_memory
):
    """Test publishing a mix of terminal and non-terminal states."""
    payload = {
        "items": [
            {
                "key": {
                    "train_id": "exp_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 0,
                },
                "value": {"done": False, "reward": 1.0, "data": {}},
            },
            {
                "key": {
                    "train_id": "exp_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 1,
                },
                "value": {"done": False, "reward": 2.0, "data": {}},
            },
            {
                "key": {
                    "train_id": "exp_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 2,
                },
                "value": {"done": True, "reward": 10.0, "data": {"reason": "success"}},
            },
        ]
    }

    response = client.post("/environment/batch/publish", json=payload)

    assert response.status_code == 200
    memory = mock_get_environment_memory

    # Verify terminal state is properly stored
    key2 = ("exp_001", 0, 1, 2)
    assert memory.get(key2).done is True
    # Verify non-terminal states
    key0 = ("exp_001", 0, 1, 0)
    key1 = ("exp_001", 0, 1, 1)
    assert memory.get(key0).done is False
    assert memory.get(key1).done is False
