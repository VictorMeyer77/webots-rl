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
    assert data["size"] == 0
    assert data["capacity"] == 100
    assert data["remaining"] == 100
    assert data["usage_percent"] == 0.0


def test_check_action_memory_health_with_data(client, mock_get_action_memory):
    """Test health check returns correct stats when memory contains data."""
    memory = mock_get_action_memory

    # Add some actions
    for i in range(5):
        memory.add(("train_001", 0, 1, i), ActionSchema(action=[float(i)]))

    response = client.get("/action/health")

    assert response.status_code == 200
    data = response.json()
    assert data["size"] == 5
    assert data["capacity"] == 100
    assert data["remaining"] == 95
    assert data["usage_percent"] == 5.0


def test_add_action_step_success(client, mock_get_action_memory):
    """Test successfully adding an action."""
    payload = {"action": [2.0]}

    response = client.post("/action/train_001/0/1/10", json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "success"}

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
            json={"action": [action_value]},
        )
        assert response.status_code == 200

    memory = mock_get_action_memory
    assert len(memory) == 4


def test_add_action_step_invalid_action_type(client, mock_get_action_memory):
    """Test validation error when action is not a list of floats."""
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
    payload = {"action": [1.0]}

    response = client.post("/action//0/1/10", json=payload)

    assert response.status_code == 404  # FastAPI returns 404 for missing path params


def test_add_action_step_negative_worker_id(client, mock_get_action_memory):
    """Test validation error for negative worker_id."""
    payload = {"action": [1.0]}

    response = client.post("/action/train_001/-1/1/10", json=payload)

    assert response.status_code == 422


def test_add_action_step_negative_episode_id(client, mock_get_action_memory):
    """Test validation error for negative episode_id."""
    payload = {"action": [1.0]}

    response = client.post("/action/train_001/0/-1/10", json=payload)

    assert response.status_code == 422


def test_add_action_step_negative_step(client, mock_get_action_memory):
    """Test validation error for negative step."""
    payload = {"action": [1.0]}

    response = client.post("/action/train_001/0/1/-1", json=payload)

    assert response.status_code == 422


def test_get_action_step_success(client, mock_get_action_memory):
    """Test successfully retrieving an action."""
    memory = mock_get_action_memory
    key = ("train_001", 0, 1, 10)
    action = ActionSchema(action=[3.0])
    memory.add(key, action)

    response = client.get("/action/train_001/0/1/10")

    assert response.status_code == 200
    data = response.json()
    assert data["action"] == [3.0]


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
    action = ActionSchema(action=[7.0])
    memory.add(key, action)

    # Retrieve multiple times
    response1 = client.get("/action/train_001/0/1/10")
    response2 = client.get("/action/train_001/0/1/10")
    response3 = client.get("/action/train_001/0/1/10")

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response3.status_code == 200
    assert response1.json()["action"] == [7.0]
    assert response2.json()["action"] == [7.0]
    assert response3.json()["action"] == [7.0]


def test_get_action_step_different_keys(client, mock_get_action_memory):
    """Test retrieving actions with different hierarchical keys."""
    memory = mock_get_action_memory

    # Add actions with different keys
    memory.add(("train_001", 0, 1, 10), ActionSchema(action=[1.0]))
    memory.add(("train_001", 1, 1, 10), ActionSchema(action=[2.0]))
    memory.add(("train_001", 0, 2, 10), ActionSchema(action=[3.0]))
    memory.add(("train_002", 0, 1, 10), ActionSchema(action=[4.0]))

    # Retrieve each one
    assert client.get("/action/train_001/0/1/10").json()["action"] == [1.0]
    assert client.get("/action/train_001/1/1/10").json()["action"] == [2.0]
    assert client.get("/action/train_001/0/2/10").json()["action"] == [3.0]
    assert client.get("/action/train_002/0/1/10").json()["action"] == [4.0]


def test_add_and_get_action_workflow(client, mock_get_action_memory):
    """Test complete workflow of adding and retrieving an action."""
    # Add action
    add_response = client.post("/action/exp_001/2/5/100", json={"action": [42.0]})
    assert add_response.status_code == 200
    assert add_response.json()["status"] == "success"

    # Retrieve action
    get_response = client.get("/action/exp_001/2/5/100")
    assert get_response.status_code == 200
    assert get_response.json()["action"] == [42.0]


def test_action_with_zero_values(client, mock_get_action_memory):
    """Test handling actions with zero as a valid action value."""
    response = client.post("/action/train_001/0/0/0", json={"action": [0.0]})
    assert response.status_code == 200

    get_response = client.get("/action/train_001/0/0/0")
    assert get_response.status_code == 200
    assert get_response.json()["action"] == [0.0]


def test_get_action_batch_all_found(client, mock_get_action_memory):
    """Test batch retrieval when all requested actions are found."""
    memory = mock_get_action_memory

    # Add actions
    memory.add(("train_001", 0, 1, 10), ActionSchema(action=[1.0]))
    memory.add(("train_001", 0, 1, 11), ActionSchema(action=[2.0]))
    memory.add(("train_001", 0, 1, 12), ActionSchema(action=[3.0]))

    # Batch request
    payload = {
        "keys": [
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 10},
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 11},
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 12},
        ]
    }

    response = client.post("/action/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert data["missing"] == 0
    assert len(data["results"]) == 3

    # Verify each result
    assert data["results"][0]["value"]["action"] == [1.0]
    assert data["results"][1]["value"]["action"] == [2.0]
    assert data["results"][2]["value"]["action"] == [3.0]


def test_get_action_batch_all_missing(client, mock_get_action_memory):
    """Test batch retrieval when none of the requested actions are found."""
    payload = {
        "keys": [
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 10},
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 11},
        ]
    }

    response = client.post("/action/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["missing"] == 2
    assert len(data["results"]) == 2

    # Verify all results are not found
    assert data["results"][0]["value"] is None
    assert data["results"][1]["value"] is None


def test_get_action_batch_mixed_results(client, mock_get_action_memory):
    """Test batch retrieval with a mix of found and missing actions."""
    memory = mock_get_action_memory

    # Add only some actions
    memory.add(("train_001", 0, 1, 10), ActionSchema(action=[1.0]))
    memory.add(("train_001", 0, 1, 12), ActionSchema(action=[3.0]))

    payload = {
        "keys": [
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 10},
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 11},
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 12},
        ]
    }

    response = client.post("/action/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert data["missing"] == 1

    # Verify specific results
    assert data["results"][0]["value"]["action"] == [1.0]
    assert data["results"][1]["value"] is None
    assert data["results"][2]["value"]["action"] == [3.0]


def test_get_action_batch_empty_keys(client, mock_get_action_memory):
    """Test batch retrieval with an empty list of keys."""
    payload = {"keys": []}

    response = client.post("/action/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["missing"] == 0
    assert len(data["results"]) == 0


def test_get_action_batch_single_key(client, mock_get_action_memory):
    """Test batch retrieval with a single key."""
    memory = mock_get_action_memory
    memory.add(("train_001", 0, 1, 10), ActionSchema(action=[5.0]))

    payload = {
        "keys": [{"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 10}]
    }

    response = client.post("/action/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["missing"] == 0
    assert data["results"][0]["value"]["action"] == [5.0]


def test_get_action_batch_different_keys(client, mock_get_action_memory):
    """Test batch retrieval with actions from different hierarchical levels."""
    memory = mock_get_action_memory

    # Add actions with different train_id, worker_id, and episode_id
    memory.add(("train_001", 0, 1, 10), ActionSchema(action=[1.0]))
    memory.add(("train_001", 1, 1, 10), ActionSchema(action=[2.0]))
    memory.add(("train_002", 0, 1, 10), ActionSchema(action=[3.0]))
    memory.add(("train_001", 0, 2, 10), ActionSchema(action=[4.0]))

    payload = {
        "keys": [
            {"train_id": "train_001", "worker_id": 0, "episode_id": 1, "step": 10},
            {"train_id": "train_001", "worker_id": 1, "episode_id": 1, "step": 10},
            {"train_id": "train_002", "worker_id": 0, "episode_id": 1, "step": 10},
            {"train_id": "train_001", "worker_id": 0, "episode_id": 2, "step": 10},
        ]
    }

    response = client.post("/action/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 4
    assert data["missing"] == 0

    # Verify each action matches
    assert data["results"][0]["value"]["action"] == [1.0]
    assert data["results"][1]["value"]["action"] == [2.0]
    assert data["results"][2]["value"]["action"] == [3.0]
    assert data["results"][3]["value"]["action"] == [4.0]


def test_get_action_batch_invalid_payload(client, mock_get_action_memory):
    """Test validation error for invalid batch request payload."""
    # Missing keys field
    payload = {}

    response = client.post("/action/batch", json=payload)

    assert response.status_code == 422


def test_get_action_batch_invalid_key_structure(client, mock_get_action_memory):
    """Test validation error for invalid key structure in batch request."""
    # Missing required fields in key
    payload = {"keys": [{"train_id": "train_001", "worker_id": 0}]}

    response = client.post("/action/batch", json=payload)

    assert response.status_code == 422


def test_get_action_batch_negative_values(client, mock_get_action_memory):
    """Test validation error for negative values in batch request keys."""
    payload = {
        "keys": [
            {"train_id": "train_001", "worker_id": -1, "episode_id": 1, "step": 10}
        ]
    }

    response = client.post("/action/batch", json=payload)

    assert response.status_code == 422


def test_action_with_large_values(client, mock_get_action_memory):
    """Test handling actions with large integer values."""
    large_action = 999999

    response = client.post("/action/train_001/0/1/10", json={"action": [large_action]})
    assert response.status_code == 200

    get_response = client.get("/action/train_001/0/1/10")
    assert get_response.status_code == 200
    assert get_response.json()["action"] == [float(large_action)]


def test_parallel_workers(client, mock_get_action_memory):
    """Test handling multiple parallel running instances."""
    num_envs = 5

    # Add actions for parallel workers
    for worker_id in range(num_envs):
        response = client.post(
            f"/action/train_001/{worker_id}/1/10", json={"action": [worker_id * 10.0]}
        )
        assert response.status_code == 200

    # Retrieve and verify each
    for worker_id in range(num_envs):
        response = client.get(f"/action/train_001/{worker_id}/1/10")
        assert response.status_code == 200
        assert response.json()["action"] == [worker_id * 10.0]


def test_episode_progression(client, mock_get_action_memory):
    """Test actions across multiple episodes."""
    for episode_id in range(3):
        for step in range(5):
            response = client.post(
                f"/action/train_001/0/{episode_id}/{step}",
                json={"action": [float(episode_id * 100 + step)]},
            )
            assert response.status_code == 200

    # Verify specific actions
    response = client.get("/action/train_001/0/1/3")
    assert response.json()["action"] == [103.0]


def test_train_id_with_special_characters(client, mock_get_action_memory):
    """Test train_id with special characters."""
    train_ids = ["train-001", "train_002", "train.003"]

    for train_id in train_ids:
        response = client.post(f"/action/{train_id}/0/1/10", json={"action": [1.0]})
        assert response.status_code == 200

        get_response = client.get(f"/action/{train_id}/0/1/10")
        assert get_response.status_code == 200


def test_action_schema_multi_value(client, mock_get_action_memory):
    """Test action schema with multiple float values (continuous action space)."""
    payload = {"action": [0.5, -1.0, 0.75]}

    response = client.post("/action/train_001/0/1/10", json=payload)
    assert response.status_code == 200

    get_response = client.get("/action/train_001/0/1/10")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["action"] == [0.5, -1.0, 0.75]


def test_action_schema_float_values(client, mock_get_action_memory):
    """Test action schema with float values."""
    payload = {"action": [3.14]}

    response = client.post("/action/train_001/0/1/10", json=payload)
    assert response.status_code == 200

    get_response = client.get("/action/train_001/0/1/10")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["action"] == [3.14]


def test_action_schema_negative_floats(client, mock_get_action_memory):
    """Test action schema with negative float values."""
    payload = {"action": [-0.5, -1.0]}

    response = client.post("/action/train_001/0/1/10", json=payload)
    assert response.status_code == 200

    get_response = client.get("/action/train_001/0/1/10")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["action"] == [-0.5, -1.0]


# Batch POST tests


def test_post_action_batch_success(client, mock_get_action_memory):
    """Test batch publishing multiple actions successfully."""
    payload = {
        "items": [
            {
                "key": {
                    "train_id": "train_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 10,
                },
                "value": {"action": [1.0]},
            },
            {
                "key": {
                    "train_id": "train_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 11,
                },
                "value": {"action": [2.0]},
            },
            {
                "key": {
                    "train_id": "train_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 12,
                },
                "value": {"action": [3.0]},
            },
        ]
    }

    response = client.post("/action/batch/publish", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total"] == 3

    # Verify all actions were stored
    get_response = client.get("/action/train_001/0/1/10")
    assert get_response.status_code == 200
    assert get_response.json()["action"] == [1.0]

    get_response = client.get("/action/train_001/0/1/11")
    assert get_response.status_code == 200
    assert get_response.json()["action"] == [2.0]

    get_response = client.get("/action/train_001/0/1/12")
    assert get_response.status_code == 200
    assert get_response.json()["action"] == [3.0]


def test_post_action_batch_empty_list(client, mock_get_action_memory):
    """Test batch publishing with empty items list."""
    payload = {"items": []}

    response = client.post("/action/batch/publish", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total"] == 0


def test_post_action_batch_single_item(client, mock_get_action_memory):
    """Test batch publishing with a single action."""
    payload = {
        "items": [
            {
                "key": {
                    "train_id": "train_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 10,
                },
                "value": {"action": [5.0]},
            }
        ]
    }

    response = client.post("/action/batch/publish", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total"] == 1

    # Verify action was stored
    get_response = client.get("/action/train_001/0/1/10")
    assert get_response.status_code == 200
    assert get_response.json()["action"] == [5.0]


def test_post_action_batch_different_keys(client, mock_get_action_memory):
    """Test batch publishing with actions from different hierarchical levels."""
    payload = {
        "items": [
            {
                "key": {
                    "train_id": "train_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 10,
                },
                "value": {"action": [1.0]},
            },
            {
                "key": {
                    "train_id": "train_001",
                    "worker_id": 1,
                    "episode_id": 1,
                    "step": 10,
                },
                "value": {"action": [2.0]},
            },
            {
                "key": {
                    "train_id": "train_002",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 10,
                },
                "value": {"action": [3.0]},
            },
            {
                "key": {
                    "train_id": "train_001",
                    "worker_id": 0,
                    "episode_id": 2,
                    "step": 10,
                },
                "value": {"action": [4.0]},
            },
        ]
    }

    response = client.post("/action/batch/publish", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total"] == 4

    # Verify all actions were stored correctly
    get_response = client.get("/action/train_001/0/1/10")
    assert get_response.json()["action"] == [1.0]

    get_response = client.get("/action/train_001/1/1/10")
    assert get_response.json()["action"] == [2.0]

    get_response = client.get("/action/train_002/0/1/10")
    assert get_response.json()["action"] == [3.0]

    get_response = client.get("/action/train_001/0/2/10")
    assert get_response.json()["action"] == [4.0]


def test_post_action_batch_invalid_payload(client, mock_get_action_memory):
    """Test batch publishing with invalid payload structure."""
    # Missing items field
    payload = {}

    response = client.post("/action/batch/publish", json=payload)

    assert response.status_code == 422


def test_post_action_batch_invalid_item_structure(client, mock_get_action_memory):
    """Test batch publishing with invalid item structure."""
    # Missing value field in item
    payload = {
        "items": [
            {
                "key": {
                    "train_id": "train_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 10,
                }
            }
        ]
    }

    response = client.post("/action/batch/publish", json=payload)

    assert response.status_code == 422


def test_post_action_batch_invalid_action(client, mock_get_action_memory):
    """Test batch publishing with invalid action value."""
    payload = {
        "items": [
            {
                "key": {
                    "train_id": "train_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 10,
                },
                "value": {"action": "invalid"},
            }
        ]
    }

    response = client.post("/action/batch/publish", json=payload)

    assert response.status_code == 422


def test_post_action_batch_negative_key_values(client, mock_get_action_memory):
    """Test batch publishing with negative values in keys."""
    payload = {
        "items": [
            {
                "key": {
                    "train_id": "train_001",
                    "worker_id": -1,
                    "episode_id": 1,
                    "step": 10,
                },
                "value": {"action": [1.0]},
            }
        ]
    }

    response = client.post("/action/batch/publish", json=payload)

    assert response.status_code == 422


def test_post_action_batch_large_batch(client, mock_get_action_memory):
    """Test batch publishing with a large number of actions."""
    items = []
    for step in range(100):
        items.append(
            {
                "key": {
                    "train_id": "train_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": step,
                },
                "value": {"action": [float(step)]},
            }
        )

    payload = {"items": items}

    response = client.post("/action/batch/publish", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total"] == 100

    # Verify a few random actions
    get_response = client.get("/action/train_001/0/1/0")
    assert get_response.json()["action"] == [0.0]

    get_response = client.get("/action/train_001/0/1/50")
    assert get_response.json()["action"] == [50.0]

    get_response = client.get("/action/train_001/0/1/99")
    assert get_response.json()["action"] == [99.0]


def test_post_action_batch_with_multi_float_actions(client, mock_get_action_memory):
    """Test batch publishing with multi-float action vectors."""
    payload = {
        "items": [
            {
                "key": {
                    "train_id": "train_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 10,
                },
                "value": {"action": [1.0, 0.5]},
            },
            {
                "key": {
                    "train_id": "train_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 11,
                },
                "value": {"action": [2.0, -0.5]},
            },
        ]
    }

    response = client.post("/action/batch/publish", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2

    get_response = client.get("/action/train_001/0/1/10")
    assert get_response.json()["action"] == [1.0, 0.5]

    get_response = client.get("/action/train_001/0/1/11")
    assert get_response.json()["action"] == [2.0, -0.5]


def test_post_action_batch_overwrite_existing(client, mock_get_action_memory):
    """Test batch publishing overwrites existing actions with same keys."""
    # First publish
    client.post("/action/train_001/0/1/10", json={"action": [1.0]})

    # Batch publish with same key
    payload = {
        "items": [
            {
                "key": {
                    "train_id": "train_001",
                    "worker_id": 0,
                    "episode_id": 1,
                    "step": 10,
                },
                "value": {"action": [99.0]},
            }
        ]
    }

    response = client.post("/action/batch/publish", json=payload)

    assert response.status_code == 200

    # Verify the action was overwritten
    get_response = client.get("/action/train_001/0/1/10")
    assert get_response.json()["action"] == [99.0]
