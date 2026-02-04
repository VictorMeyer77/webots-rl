"""Unit tests for root-level router endpoints."""

import pytest
from app.core.memory import Memory
from app.core.supervisor import Supervisor
from app.dependencies import (
    get_action_memory,
    get_environment_memory,
    get_observation_memory,
    get_supervisor,
)
from app.routers import router
from app.schemas import SystemInfoSchema
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    """Create a FastAPI application with the root router.

    Returns:
        FastAPI: Configured application instance for testing.
    """
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    """Create a test client for the FastAPI application.

    Args:
        app: FastAPI application instance.

    Returns:
        TestClient: Client for making HTTP requests to the application.
    """
    return TestClient(app)


@pytest.fixture
def supervisor():
    """Provide a fresh supervisor instance for each test.

    Returns:
        Supervisor: New supervisor instance for managing training sessions.
    """
    return Supervisor()


@pytest.fixture
def action_memory():
    """Provide a fresh action memory instance for each test.

    Returns:
        Memory: New memory instance for storing actions.
    """
    return Memory(100)


@pytest.fixture
def observation_memory():
    """Provide a fresh observation memory instance for each test.

    Returns:
        Memory: New memory instance for storing observations.
    """
    return Memory(100)


@pytest.fixture
def environment_memory():
    """Provide a fresh environment memory instance for each test.

    Returns:
        Memory: New memory instance for storing environment states.
    """
    return Memory(100)


@pytest.fixture
def override_dependencies(
    app, supervisor, action_memory, observation_memory, environment_memory
):
    """Override all dependencies with test instances.

    Args:
        app: FastAPI application instance.
        supervisor: Test supervisor instance.
        action_memory: Test action memory instance.
        observation_memory: Test observation memory instance.
        environment_memory: Test environment memory instance.

    Yields:
        dict: Dictionary containing all test component instances.
    """
    app.dependency_overrides[get_supervisor] = lambda: supervisor
    app.dependency_overrides[get_action_memory] = lambda: action_memory
    app.dependency_overrides[get_observation_memory] = lambda: observation_memory
    app.dependency_overrides[get_environment_memory] = lambda: environment_memory

    yield {
        "supervisor": supervisor,
        "action_memory": action_memory,
        "observation_memory": observation_memory,
        "environment_memory": environment_memory,
    }

    app.dependency_overrides.clear()


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check_returns_200(self, client, override_dependencies):
        """Test that health check returns 200 status code."""
        response = client.get("/health")

        assert response.status_code == 200

    def test_health_check_validates_against_schema(self, client, override_dependencies):
        """Test that health check response validates against SystemInfoSchema."""
        response = client.get("/health")
        data = response.json()

        system_info = SystemInfoSchema(**data)
        assert system_info is not None

    def test_health_check_schema_structure(self, client, override_dependencies):
        """Test that health check returns data matching SystemInfoSchema structure."""
        deps = override_dependencies
        deps["supervisor"].add_train("train_001")
        _ = deps["supervisor"].add_worker("train_001")

        response = client.get("/health")
        data = response.json()

        assert "supervisor" in data
        assert "action_memory" in data
        assert "observation_memory" in data
        assert "environment_memory" in data

        assert isinstance(data["supervisor"], dict)
        for train_id, workers in data["supervisor"].items():
            assert isinstance(train_id, str)
            assert isinstance(workers, dict)
            for worker_id, episode_id in workers.items():
                assert isinstance(worker_id, str)
                assert worker_id.isdigit()
                assert isinstance(episode_id, int)

        for memory_key in ["action_memory", "observation_memory", "environment_memory"]:
            assert isinstance(data[memory_key], dict)
            for key, value in data[memory_key].items():
                assert isinstance(key, str)
                assert isinstance(value, (int, float))

    def test_health_check_empty_state(self, client, override_dependencies):
        """Test health check with empty system state validates against schema."""
        response = client.get("/health")
        data = response.json()

        system_info = SystemInfoSchema(**data)
        assert system_info.supervisor == {}
        assert isinstance(system_info.action_memory, dict)
        assert isinstance(system_info.observation_memory, dict)
        assert isinstance(system_info.environment_memory, dict)

    def test_health_check_with_supervisor_data(self, client, override_dependencies):
        """Test health check includes supervisor training data matching schema."""
        deps = override_dependencies
        deps["supervisor"].add_train("train_001")
        worker_id = deps["supervisor"].add_worker("train_001")

        response = client.get("/health")
        data = response.json()

        system_info = SystemInfoSchema(**data)
        assert "train_001" in system_info.supervisor
        assert worker_id in system_info.supervisor["train_001"]
        assert system_info.supervisor["train_001"][worker_id] == 0

    def test_health_check_with_multiple_trains(self, client, override_dependencies):
        """Test health check with multiple training sessions validates against schema."""
        deps = override_dependencies
        deps["supervisor"].add_train("train_001")
        deps["supervisor"].add_train("train_002")
        deps["supervisor"].add_worker("train_001")
        deps["supervisor"].add_worker("train_002")

        response = client.get("/health")
        data = response.json()

        system_info = SystemInfoSchema(**data)
        assert len(system_info.supervisor) == 2
        assert "train_001" in system_info.supervisor
        assert "train_002" in system_info.supervisor

    def test_health_check_with_incremented_episodes(
        self, client, override_dependencies
    ):
        """Test health check reflects incremented episode IDs in schema."""
        deps = override_dependencies
        deps["supervisor"].add_train("train_001")
        worker_id = deps["supervisor"].add_worker("train_001")

        for _ in range(5):
            deps["supervisor"].increment_episode_id("train_001", worker_id)

        response = client.get("/health")
        data = response.json()

        system_info = SystemInfoSchema(**data)
        assert system_info.supervisor["train_001"][worker_id] == 5

    def test_health_check_reflects_current_state(self, client, override_dependencies):
        """Test that health check always reflects the current system state in schema."""
        deps = override_dependencies
        deps["supervisor"].add_train("train_001")
        worker_id = deps["supervisor"].add_worker("train_001")

        response1 = client.get("/health")
        data1 = SystemInfoSchema(**response1.json())
        assert data1.supervisor["train_001"][worker_id] == 0

        deps["supervisor"].increment_episode_id("train_001", worker_id)

        response2 = client.get("/health")
        data2 = SystemInfoSchema(**response2.json())
        assert data2.supervisor["train_001"][worker_id] == 1

    def test_health_check_with_multiple_workers(self, client, override_dependencies):
        """Test health check with multiple workers in same training session."""
        deps = override_dependencies
        deps["supervisor"].add_train("train_001")
        w1 = deps["supervisor"].add_worker("train_001")
        w2 = deps["supervisor"].add_worker("train_001")
        w3 = deps["supervisor"].add_worker("train_001")

        for _ in range(3):
            deps["supervisor"].increment_episode_id("train_001", w1)

        for _ in range(7):
            deps["supervisor"].increment_episode_id("train_001", w2)

        response = client.get("/health")
        data = response.json()

        system_info = SystemInfoSchema(**data)
        assert len(system_info.supervisor["train_001"]) == 3
        assert system_info.supervisor["train_001"][w1] == 3
        assert system_info.supervisor["train_001"][w2] == 7
        assert system_info.supervisor["train_001"][w3] == 0

    def test_health_check_memory_stats_match_schema(
        self, client, override_dependencies
    ):
        """Test that memory stats match SystemInfoSchema type constraints."""
        response = client.get("/health")
        data = response.json()

        system_info = SystemInfoSchema(**data)

        for memory_attr in [
            system_info.action_memory,
            system_info.observation_memory,
            system_info.environment_memory,
        ]:
            assert isinstance(memory_attr, dict)
            for key, value in memory_attr.items():
                assert isinstance(key, str)
                assert isinstance(value, (int, float))

    def test_health_check_json_serializable(self, client, override_dependencies):
        """Test that health check response is JSON serializable and validates."""
        deps = override_dependencies
        deps["supervisor"].add_train("train_001")
        deps["supervisor"].add_worker("train_001")

        response = client.get("/health")
        data = response.json()

        assert data is not None

        system_info = SystemInfoSchema(**data)
        assert system_info is not None

    def test_health_check_method_is_get(self, client, override_dependencies):
        """Test that health check only accepts GET requests."""
        response_post = client.post("/health")
        response_put = client.put("/health")
        response_delete = client.delete("/health")

        assert response_post.status_code == 405
        assert response_put.status_code == 405
        assert response_delete.status_code == 405

    def test_health_check_integration(self, client, override_dependencies):
        """Test complete health check workflow with all components validating against schema."""
        deps = override_dependencies

        deps["supervisor"].add_train("exp_001")
        deps["supervisor"].add_train("exp_002")

        w1 = deps["supervisor"].add_worker("exp_001")
        w2 = deps["supervisor"].add_worker("exp_001")
        w3 = deps["supervisor"].add_worker("exp_002")

        for _ in range(10):
            deps["supervisor"].increment_episode_id("exp_001", w1)

        for _ in range(5):
            deps["supervisor"].increment_episode_id("exp_001", w2)

        for _ in range(15):
            deps["supervisor"].increment_episode_id("exp_002", w3)

        response = client.get("/health")
        data = response.json()

        system_info = SystemInfoSchema(**data)

        assert response.status_code == 200
        assert len(system_info.supervisor) == 2

        assert system_info.supervisor["exp_001"][w1] == 10
        assert system_info.supervisor["exp_001"][w2] == 5
        assert system_info.supervisor["exp_002"][w3] == 15

        assert system_info.action_memory is not None
        assert system_info.observation_memory is not None
        assert system_info.environment_memory is not None


class TestHealthEndpointDocumentation:
    """Tests for health endpoint documentation and metadata."""

    def test_health_endpoint_has_summary(self, app):
        """Test that health endpoint has a summary."""
        routes = [
            route
            for route in app.routes
            if hasattr(route, "path") and route.path == "/health"
        ]
        assert len(routes) == 1
        assert routes[0].summary == "System health check"

    def test_health_endpoint_has_tags(self, app):
        """Test that health endpoint is tagged correctly."""
        routes = [
            route
            for route in app.routes
            if hasattr(route, "path") and route.path == "/health"
        ]
        assert len(routes) == 1
        assert "Health" in routes[0].tags

    def test_health_endpoint_has_description(self, app):
        """Test that health endpoint has a description."""
        routes = [
            route
            for route in app.routes
            if hasattr(route, "path") and route.path == "/health"
        ]
        assert len(routes) == 1
        assert routes[0].description is not None
        assert len(routes[0].description) > 0

    def test_health_endpoint_has_response_model(self, app):
        """Test that health endpoint declares SystemInfoSchema as response model."""
        routes = [
            route
            for route in app.routes
            if hasattr(route, "path") and route.path == "/health"
        ]
        assert len(routes) == 1
        assert routes[0].response_model == SystemInfoSchema

    def test_health_endpoint_has_status_code(self, app):
        """Test that health endpoint declares 200 status code."""
        routes = [
            route
            for route in app.routes
            if hasattr(route, "path") and route.path == "/health"
        ]
        assert len(routes) == 1
        assert 200 in routes[0].responses

    def test_health_endpoint_has_example_response(self, app):
        """Test that health endpoint has example response in documentation."""
        routes = [
            route
            for route in app.routes
            if hasattr(route, "path") and route.path == "/health"
        ]
        assert len(routes) == 1
        assert 200 in routes[0].responses
        assert "content" in routes[0].responses[200]
        assert "application/json" in routes[0].responses[200]["content"]
        assert "example" in routes[0].responses[200]["content"]["application/json"]


class TestSystemInfoSchema:
    """Tests for SystemInfoSchema validation."""

    def test_schema_with_valid_data(self):
        """Test SystemInfoSchema with valid data."""
        data = {
            "supervisor": {"train_001": {0: 5, 1: 10}},
            "action_memory": {"size": 100, "capacity": 1000, "usage": 0.1},
            "observation_memory": {"size": 200, "capacity": 1000, "usage": 0.2},
            "environment_memory": {"size": 150, "capacity": 1000, "usage": 0.15},
        }

        schema = SystemInfoSchema(**data)
        assert schema.supervisor == {"train_001": {0: 5, 1: 10}}
        assert schema.action_memory["size"] == 100
        assert schema.observation_memory["size"] == 200
        assert schema.environment_memory["size"] == 150

    def test_schema_with_empty_supervisor(self):
        """Test SystemInfoSchema with empty supervisor data."""
        data = {
            "supervisor": {},
            "action_memory": {},
            "observation_memory": {},
            "environment_memory": {},
        }

        schema = SystemInfoSchema(**data)
        assert schema.supervisor == {}

    def test_schema_requires_all_fields(self):
        """Test that SystemInfoSchema requires all fields."""
        with pytest.raises(Exception):
            SystemInfoSchema(supervisor={})

    def test_schema_validates_supervisor_types(self):
        """Test that SystemInfoSchema validates supervisor type structure."""
        valid_data = {
            "supervisor": {"train_001": {0: 5}},
            "action_memory": {},
            "observation_memory": {},
            "environment_memory": {},
        }

        schema = SystemInfoSchema(**valid_data)
        assert isinstance(schema.supervisor["train_001"][0], int)

    def test_schema_validates_memory_types(self):
        """Test that SystemInfoSchema validates memory type structure."""
        data = {
            "supervisor": {},
            "action_memory": {"size": 100, "usage": 0.5},
            "observation_memory": {"size": 200, "capacity": 1000},
            "environment_memory": {"items": 50},
        }

        schema = SystemInfoSchema(**data)
        assert isinstance(schema.action_memory["size"], int)
        assert isinstance(schema.action_memory["usage"], float)

    def test_schema_allows_int_and_float_in_memory(self):
        """Test that SystemInfoSchema accepts both int and float values in memory dicts."""
        data = {
            "supervisor": {},
            "action_memory": {"int_value": 100, "float_value": 10.5},
            "observation_memory": {"mixed": 50, "percent": 0.75},
            "environment_memory": {"count": 25, "ratio": 1.5},
        }

        schema = SystemInfoSchema(**data)
        assert schema.action_memory["int_value"] == 100
        assert schema.action_memory["float_value"] == 10.5
        assert isinstance(schema.observation_memory["mixed"], int)
        assert isinstance(schema.observation_memory["percent"], float)
