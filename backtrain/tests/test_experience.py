"""Unit tests for the Experience Tuple Communication Router."""

import pytest
from app.core.memory import Memory
from app.dependencies import (
    get_action_memory,
    get_environment_memory,
    get_observation_memory,
)
from app.routers.experience import router
from app.schemas import ActionSchema, EnvironmentSchema, ObservationSchema
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    """Create a FastAPI application with the experience router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def environment_memory():
    """Provide a fresh environment memory instance for each test."""
    return Memory(capacity=100)


@pytest.fixture
def observation_memory():
    """Provide a fresh observation memory instance for each test."""
    return Memory(capacity=100)


@pytest.fixture
def action_memory():
    """Provide a fresh action memory instance for each test."""
    return Memory(capacity=100)


@pytest.fixture
def mock_memories(environment_memory, observation_memory, action_memory, app):
    """Mock all memory dependencies."""

    def _get_env_memory():
        return environment_memory

    def _get_obs_memory():
        return observation_memory

    def _get_action_memory():
        return action_memory

    # Override the dependencies in the app
    app.dependency_overrides[get_environment_memory] = _get_env_memory
    app.dependency_overrides[get_observation_memory] = _get_obs_memory
    app.dependency_overrides[get_action_memory] = _get_action_memory

    yield {
        "environment": environment_memory,
        "observation": observation_memory,
        "action": action_memory,
    }

    # Clean up after test
    app.dependency_overrides.clear()


@pytest.fixture
def client(app, mock_memories):
    """Create a test client for the FastAPI application."""
    return TestClient(app)


def test_get_experience_step_success_non_terminal(client, mock_memories):
    """Test successfully retrieving a complete non-terminal experience tuple."""
    key = ("train_001", 0, 1, 10)
    next_key = ("train_001", 0, 1, 11)

    # Add all required components
    mock_memories["environment"].add(
        key, EnvironmentSchema(done=False, reward=1.5, data={"info": "step"})
    )
    mock_memories["observation"].add(
        key, ObservationSchema(data={"position": [1.0, 2.0]})
    )
    mock_memories["action"].add(key, ActionSchema(action=2))
    mock_memories["observation"].add(
        next_key, ObservationSchema(data={"position": [1.5, 2.5]})
    )

    response = client.get("/experience/train_001/0/1/10")

    assert response.status_code == 200
    data = response.json()
    assert data["observation"]["data"]["position"] == [1.0, 2.0]
    assert data["action"]["action"] == 2
    assert data["environment"]["done"] is False
    assert data["environment"]["reward"] == 1.5
    assert data["next_observation"]["data"]["position"] == [1.5, 2.5]


def test_get_experience_step_success_terminal(client, mock_memories):
    """Test successfully retrieving a complete terminal experience tuple."""
    key = ("train_001", 0, 1, 100)

    # Add all required components (no next_observation for terminal state)
    mock_memories["environment"].add(
        key, EnvironmentSchema(done=True, reward=10.0, data={"final": True})
    )
    mock_memories["observation"].add(
        key, ObservationSchema(data={"position": [10.0, 10.0]})
    )
    mock_memories["action"].add(key, ActionSchema(action=3))

    response = client.get("/experience/train_001/0/1/100")

    assert response.status_code == 200
    data = response.json()
    assert data["observation"]["data"]["position"] == [10.0, 10.0]
    assert data["action"]["action"] == 3
    assert data["environment"]["done"] is True
    assert data["environment"]["reward"] == 10.0
    assert data["next_observation"] is None


def test_get_experience_step_missing_environment(client, mock_memories):
    """Test 404 error when environment state is missing."""
    key = ("train_001", 0, 1, 10)
    next_key = ("train_001", 0, 1, 11)

    # Add observation, action, next_observation but not environment
    mock_memories["observation"].add(
        key, ObservationSchema(data={"position": [1.0, 2.0]})
    )
    mock_memories["action"].add(key, ActionSchema(action=2))
    mock_memories["observation"].add(
        next_key, ObservationSchema(data={"position": [1.5, 2.5]})
    )

    response = client.get("/experience/train_001/0/1/10")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "Experience not found" in data["detail"]


def test_get_experience_step_missing_observation(client, mock_memories):
    """Test 404 error when observation is missing."""
    key = ("train_001", 0, 1, 10)
    next_key = ("train_001", 0, 1, 11)

    # Add environment, action, next_observation but not observation
    mock_memories["environment"].add(
        key, EnvironmentSchema(done=False, reward=1.5, data={})
    )
    mock_memories["action"].add(key, ActionSchema(action=2))
    mock_memories["observation"].add(
        next_key, ObservationSchema(data={"position": [1.5, 2.5]})
    )

    response = client.get("/experience/train_001/0/1/10")

    assert response.status_code == 404


def test_get_experience_step_missing_action(client, mock_memories):
    """Test 404 error when action is missing."""
    key = ("train_001", 0, 1, 10)
    next_key = ("train_001", 0, 1, 11)

    # Add environment, observation, next_observation but not action
    mock_memories["environment"].add(
        key, EnvironmentSchema(done=False, reward=1.5, data={})
    )
    mock_memories["observation"].add(
        key, ObservationSchema(data={"position": [1.0, 2.0]})
    )
    mock_memories["observation"].add(
        next_key, ObservationSchema(data={"position": [1.5, 2.5]})
    )

    response = client.get("/experience/train_001/0/1/10")

    assert response.status_code == 404


def test_get_experience_step_missing_next_observation_non_terminal(
    client, mock_memories
):
    """Test 404 error when next_observation is missing for non-terminal state."""
    key = ("train_001", 0, 1, 10)

    # Add environment (done=False), observation, action but not next_observation
    mock_memories["environment"].add(
        key, EnvironmentSchema(done=False, reward=1.5, data={})
    )
    mock_memories["observation"].add(
        key, ObservationSchema(data={"position": [1.0, 2.0]})
    )
    mock_memories["action"].add(key, ActionSchema(action=2))

    response = client.get("/experience/train_001/0/1/10")

    assert response.status_code == 404


def test_get_experience_step_all_components_missing(client, mock_memories):
    """Test 404 error when all components are missing."""
    response = client.get("/experience/train_001/0/1/10")

    assert response.status_code == 404


def test_get_experience_step_multiple_retrievals(client, mock_memories):
    """Test that get is non-destructive and experience can be retrieved multiple times."""
    key = ("train_001", 0, 1, 10)

    mock_memories["environment"].add(
        key, EnvironmentSchema(done=True, reward=5.0, data={})
    )
    mock_memories["observation"].add(key, ObservationSchema(data={"state": "final"}))
    mock_memories["action"].add(key, ActionSchema(action=1))

    # Retrieve multiple times
    response1 = client.get("/experience/train_001/0/1/10")
    response2 = client.get("/experience/train_001/0/1/10")
    response3 = client.get("/experience/train_001/0/1/10")

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response3.status_code == 200
    assert response1.json()["environment"]["reward"] == 5.0
    assert response2.json()["environment"]["reward"] == 5.0
    assert response3.json()["environment"]["reward"] == 5.0


def test_get_experience_step_different_keys(client, mock_memories):
    """Test retrieving experiences with different hierarchical keys."""
    keys = [
        ("train_001", 0, 1, 10),
        ("train_001", 1, 1, 10),
        ("train_001", 0, 2, 10),
        ("train_002", 0, 1, 10),
    ]

    # Add experiences for each key
    for i, key in enumerate(keys):
        mock_memories["environment"].add(
            key, EnvironmentSchema(done=True, reward=float(i), data={})
        )
        mock_memories["observation"].add(key, ObservationSchema(data={"id": i}))
        mock_memories["action"].add(key, ActionSchema(action=i))

    # Retrieve each one
    response1 = client.get("/experience/train_001/0/1/10")
    assert response1.json()["environment"]["reward"] == 0.0

    response2 = client.get("/experience/train_001/1/1/10")
    assert response2.json()["environment"]["reward"] == 1.0

    response3 = client.get("/experience/train_001/0/2/10")
    assert response3.json()["environment"]["reward"] == 2.0

    response4 = client.get("/experience/train_002/0/1/10")
    assert response4.json()["environment"]["reward"] == 3.0


def test_get_experience_step_invalid_train_id(client, mock_memories):
    """Test validation error for empty train_id."""
    response = client.get("/experience//0/1/10")

    assert response.status_code == 404


def test_get_experience_step_negative_env_id(client, mock_memories):
    """Test validation error for negative env_id."""
    response = client.get("/experience/train_001/-1/1/10")

    assert response.status_code == 422


def test_get_experience_step_negative_episode_id(client, mock_memories):
    """Test validation error for negative episode_id."""
    response = client.get("/experience/train_001/0/-1/10")

    assert response.status_code == 422


def test_get_experience_step_negative_step(client, mock_memories):
    """Test validation error for negative step."""
    response = client.get("/experience/train_001/0/1/-1")

    assert response.status_code == 422


def test_get_experience_step_episode_sequence(client, mock_memories):
    """Test retrieving experiences from a sequence of episode steps."""
    train_id = "train_001"
    env_id = 0
    episode_id = 1

    # Create a sequence of 5 steps
    for step in range(5):
        key = (train_id, env_id, episode_id, step)
        next_key = (train_id, env_id, episode_id, step + 1)

        is_terminal = step == 4

        mock_memories["environment"].add(
            key,
            EnvironmentSchema(
                done=is_terminal, reward=float(step), data={"step": step}
            ),
        )
        mock_memories["observation"].add(
            key, ObservationSchema(data={"position": step})
        )
        mock_memories["action"].add(key, ActionSchema(action=step))

        # Add next observation for non-terminal states
        if not is_terminal:
            mock_memories["observation"].add(
                next_key, ObservationSchema(data={"position": step + 1})
            )

    # Verify each step
    for step in range(5):
        response = client.get(f"/experience/{train_id}/{env_id}/{episode_id}/{step}")
        assert response.status_code == 200
        data = response.json()
        assert data["observation"]["data"]["position"] == step
        assert data["action"]["action"] == step
        assert data["environment"]["reward"] == float(step)

        if step < 4:
            assert data["next_observation"]["data"]["position"] == step + 1
        else:
            assert data["next_observation"] is None


def test_get_experience_step_parallel_environments(client, mock_memories):
    """Test retrieving experiences from multiple parallel environments."""
    num_envs = 5
    train_id = "train_001"
    episode_id = 1
    step = 10

    # Add experiences for each parallel environment
    for env_id in range(num_envs):
        key = (train_id, env_id, episode_id, step)

        mock_memories["environment"].add(
            key,
            EnvironmentSchema(
                done=True, reward=float(env_id * 10), data={"env_id": env_id}
            ),
        )
        mock_memories["observation"].add(key, ObservationSchema(data={"env": env_id}))
        mock_memories["action"].add(key, ActionSchema(action=env_id))

    # Retrieve and verify each environment's experience
    for env_id in range(num_envs):
        response = client.get(f"/experience/{train_id}/{env_id}/{episode_id}/{step}")
        assert response.status_code == 200
        data = response.json()
        assert data["environment"]["reward"] == env_id * 10
        assert data["observation"]["data"]["env"] == env_id
        assert data["action"]["action"] == env_id


def test_get_experience_step_complex_observation_data(client, mock_memories):
    """Test handling experiences with complex nested observation data."""
    key = ("train_001", 0, 1, 10)

    complex_obs_data = {
        "sensors": [0.1, 0.2, 0.3, 0.4, 0.5],
        "camera": {"width": 640, "height": 480, "pixels": [[1, 2], [3, 4]]},
        "metadata": {"timestamp": 123456789, "agent_id": "agent_001"},
    }

    mock_memories["environment"].add(
        key, EnvironmentSchema(done=True, reward=5.0, data={})
    )
    mock_memories["observation"].add(key, ObservationSchema(data=complex_obs_data))
    mock_memories["action"].add(key, ActionSchema(action=3))

    response = client.get("/experience/train_001/0/1/10")

    assert response.status_code == 200
    data = response.json()
    assert data["observation"]["data"]["sensors"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert data["observation"]["data"]["camera"]["width"] == 640
    assert data["observation"]["data"]["metadata"]["agent_id"] == "agent_001"


def test_get_experience_step_complex_environment_data(client, mock_memories):
    """Test handling experiences with complex environment data."""
    key = ("train_001", 0, 1, 10)

    complex_env_data = {
        "collision": True,
        "objects_collected": ["coin", "powerup"],
        "score_breakdown": {"base": 10, "bonus": 5, "penalty": -2},
    }

    mock_memories["environment"].add(
        key, EnvironmentSchema(done=True, reward=13.0, data=complex_env_data)
    )
    mock_memories["observation"].add(key, ObservationSchema(data={"state": "final"}))
    mock_memories["action"].add(key, ActionSchema(action=1))

    response = client.get("/experience/train_001/0/1/10")

    assert response.status_code == 200
    data = response.json()
    assert data["environment"]["data"]["collision"] is True
    assert data["environment"]["data"]["objects_collected"] == ["coin", "powerup"]
    assert data["environment"]["data"]["score_breakdown"]["bonus"] == 5


def test_get_experience_step_zero_reward(client, mock_memories):
    """Test handling experiences with zero reward."""
    key = ("train_001", 0, 1, 10)

    mock_memories["environment"].add(
        key, EnvironmentSchema(done=True, reward=0.0, data={})
    )
    mock_memories["observation"].add(key, ObservationSchema(data={"state": "neutral"}))
    mock_memories["action"].add(key, ActionSchema(action=0))

    response = client.get("/experience/train_001/0/1/10")

    assert response.status_code == 200
    assert response.json()["environment"]["reward"] == 0.0


def test_get_experience_step_negative_reward(client, mock_memories):
    """Test handling experiences with negative reward."""
    key = ("train_001", 0, 1, 10)

    mock_memories["environment"].add(
        key, EnvironmentSchema(done=False, reward=-5.0, data={"penalty": True})
    )
    mock_memories["observation"].add(key, ObservationSchema(data={"state": "bad"}))
    mock_memories["action"].add(key, ActionSchema(action=2))
    mock_memories["observation"].add(
        ("train_001", 0, 1, 11), ObservationSchema(data={"state": "recovering"})
    )

    response = client.get("/experience/train_001/0/1/10")

    assert response.status_code == 200
    assert response.json()["environment"]["reward"] == -5.0


def test_get_experience_step_train_id_with_special_characters(client, mock_memories):
    """Test train_id with special characters."""
    train_ids = ["train-001", "train_002", "train.003"]

    for train_id in train_ids:
        key = (train_id, 0, 1, 10)

        mock_memories["environment"].add(
            key, EnvironmentSchema(done=True, reward=1.0, data={})
        )
        mock_memories["observation"].add(key, ObservationSchema(data={"id": train_id}))
        mock_memories["action"].add(key, ActionSchema(action=1))

        response = client.get(f"/experience/{train_id}/0/1/10")
        assert response.status_code == 200
        assert response.json()["observation"]["data"]["id"] == train_id


def test_get_experience_step_after_partial_eviction(app):
    """Test that partially evicted experiences return 404."""
    # Create small memories
    small_env_memory = Memory(capacity=2)
    small_obs_memory = Memory(capacity=100)
    small_action_memory = Memory(capacity=100)

    # Override dependencies
    app.dependency_overrides[get_environment_memory] = lambda: small_env_memory
    app.dependency_overrides[get_observation_memory] = lambda: small_obs_memory
    app.dependency_overrides[get_action_memory] = lambda: small_action_memory

    client = TestClient(app)

    # Add three experiences to trigger environment eviction
    for step in range(3):
        key = ("train_001", 0, 1, step)
        small_env_memory.add(
            key, EnvironmentSchema(done=False, reward=float(step), data={})
        )
        small_obs_memory.add(key, ObservationSchema(data={"step": step}))
        small_action_memory.add(key, ActionSchema(action=step))
        small_obs_memory.add(
            ("train_001", 0, 1, step + 1), ObservationSchema(data={"step": step + 1})
        )

    # First experience should have evicted environment
    response = client.get("/experience/train_001/0/1/0")
    assert response.status_code == 404

    # Later experiences should still work
    response = client.get("/experience/train_001/0/1/1")
    assert response.status_code == 200

    # Clean up
    app.dependency_overrides.clear()


def test_get_experience_step_multiple_episodes(client, mock_memories):
    """Test retrieving experiences across multiple episodes."""
    train_id = "train_001"
    env_id = 0

    for episode_id in range(3):
        for step in range(3):
            key = (train_id, env_id, episode_id, step)

            is_terminal = step == 2

            mock_memories["environment"].add(
                key,
                EnvironmentSchema(
                    done=is_terminal,
                    reward=float(episode_id * 10 + step),
                    data={},
                ),
            )
            mock_memories["observation"].add(
                key, ObservationSchema(data={"episode": episode_id, "step": step})
            )
            mock_memories["action"].add(key, ActionSchema(action=step))

            if not is_terminal:
                mock_memories["observation"].add(
                    (train_id, env_id, episode_id, step + 1),
                    ObservationSchema(data={"episode": episode_id, "step": step + 1}),
                )

    # Verify experiences from different episodes
    response1 = client.get(f"/experience/{train_id}/{env_id}/0/1")
    assert response1.status_code == 200
    assert response1.json()["environment"]["reward"] == 1.0

    response2 = client.get(f"/experience/{train_id}/{env_id}/1/1")
    assert response2.status_code == 200
    assert response2.json()["environment"]["reward"] == 11.0

    response3 = client.get(f"/experience/{train_id}/{env_id}/2/1")
    assert response3.status_code == 200
    assert response3.json()["environment"]["reward"] == 21.0
