"""Unit tests for the router initialization module.

This module tests the root-level API endpoints, specifically the health check
endpoint that provides system status and statistics.
"""

from unittest.mock import Mock

import pytest
from app.core.memory import Memory
from app.core.supervisor import Supervisor
from app.schemas.health import MemoryStatsSchema, SystemInfoSchema
from app.schemas.supervisor import TrainingSchema, WorkerSchema
from fastapi.testclient import TestClient


@pytest.fixture
def mock_supervisor():
    """Create a mock Supervisor instance with sample training data."""
    supervisor = Mock(spec=Supervisor)
    supervisor.trainings = [
        TrainingSchema(
            id="train_001",
            workers=[
                WorkerSchema(id=0, episode_id=42, status=True),
                WorkerSchema(id=1, episode_id=38, status=False),
            ],
        ),
        TrainingSchema(
            id="train_002",
            workers=[
                WorkerSchema(id=0, episode_id=10, status=True),
            ],
        ),
    ]
    return supervisor


@pytest.fixture
def mock_action_memory():
    """Create a mock Memory instance for action channel."""
    memory = Mock(spec=Memory)
    memory.stats.return_value = MemoryStatsSchema(
        size=150,
        capacity=1000,
        remaining=850,
        usage_percent=15.0,
    )
    return memory


@pytest.fixture
def mock_observation_memory():
    """Create a mock Memory instance for observation channel."""
    memory = Mock(spec=Memory)
    memory.stats.return_value = MemoryStatsSchema(
        size=200,
        capacity=1000,
        remaining=800,
        usage_percent=20.0,
    )
    return memory


@pytest.fixture
def mock_environment_memory():
    """Create a mock Memory instance for environment channel."""
    memory = Mock(spec=Memory)
    memory.stats.return_value = MemoryStatsSchema(
        size=100,
        capacity=1000,
        remaining=900,
        usage_percent=10.0,
    )
    return memory


@pytest.fixture
def app_with_mocks(
    mock_supervisor,
    mock_action_memory,
    mock_observation_memory,
    mock_environment_memory,
):
    """Create a FastAPI application with mocked dependencies."""
    from app.dependencies import (
        get_action_memory,
        get_environment_memory,
        get_observation_memory,
        get_supervisor,
    )
    from app.routers import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    # Override dependencies
    app.dependency_overrides[get_supervisor] = lambda: mock_supervisor
    app.dependency_overrides[get_action_memory] = lambda: mock_action_memory
    app.dependency_overrides[get_observation_memory] = lambda: mock_observation_memory
    app.dependency_overrides[get_environment_memory] = lambda: mock_environment_memory

    return app


class TestHealthEndpoint:
    """Test suite for the /health endpoint."""

    def test_health_endpoint_returns_200(self, app_with_mocks):
        """Test that the health endpoint returns a 200 status code."""
        client = TestClient(app_with_mocks)
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_endpoint_returns_correct_structure(self, app_with_mocks):
        """Test that the health endpoint returns data with the correct structure."""
        client = TestClient(app_with_mocks)
        response = client.get("/health")
        data = response.json()

        assert "supervisor" in data
        assert "action_memory" in data
        assert "observation_memory" in data
        assert "environment_memory" in data

    def test_health_endpoint_supervisor_data(self, app_with_mocks, mock_supervisor):
        """Test that the health endpoint returns correct supervisor data."""
        client = TestClient(app_with_mocks)
        response = client.get("/health")
        data = response.json()

        assert len(data["supervisor"]) == 2
        assert data["supervisor"][0]["id"] == "train_001"
        assert len(data["supervisor"][0]["workers"]) == 2
        assert data["supervisor"][0]["workers"][0]["id"] == 0
        assert data["supervisor"][0]["workers"][0]["episode_id"] == 42
        assert data["supervisor"][0]["workers"][0]["status"] is True

    def test_health_endpoint_memory_stats(self, app_with_mocks):
        """Test that the health endpoint returns correct memory statistics."""
        client = TestClient(app_with_mocks)
        response = client.get("/health")
        data = response.json()

        # Check action memory
        assert data["action_memory"]["size"] == 150
        assert data["action_memory"]["capacity"] == 1000
        assert data["action_memory"]["remaining"] == 850
        assert data["action_memory"]["usage_percent"] == 15.0

        # Check observation memory
        assert data["observation_memory"]["size"] == 200
        assert data["observation_memory"]["capacity"] == 1000
        assert data["observation_memory"]["remaining"] == 800
        assert data["observation_memory"]["usage_percent"] == 20.0

        # Check environment memory
        assert data["environment_memory"]["size"] == 100
        assert data["environment_memory"]["capacity"] == 1000
        assert data["environment_memory"]["remaining"] == 900
        assert data["environment_memory"]["usage_percent"] == 10.0

    def test_health_endpoint_calls_stats_methods(
        self,
        app_with_mocks,
        mock_action_memory,
        mock_observation_memory,
        mock_environment_memory,
    ):
        """Test that the health endpoint calls stats() on all memory instances."""
        client = TestClient(app_with_mocks)
        client.get("/health")

        mock_action_memory.stats.assert_called_once()
        mock_observation_memory.stats.assert_called_once()
        mock_environment_memory.stats.assert_called_once()

    def test_health_endpoint_empty_supervisor(self, app_with_mocks, mock_supervisor):
        """Test the health endpoint with no active training sessions."""
        mock_supervisor.trainings = []
        client = TestClient(app_with_mocks)
        response = client.get("/health")
        data = response.json()

        assert response.status_code == 200
        assert data["supervisor"] == []

    def test_health_endpoint_response_model_validation(self, app_with_mocks):
        """Test that the health endpoint response validates against SystemInfoSchema."""
        client = TestClient(app_with_mocks)
        response = client.get("/health")
        data = response.json()

        # Validate that the response can be parsed into SystemInfoSchema
        system_info = SystemInfoSchema(**data)
        assert system_info is not None
        assert isinstance(system_info.supervisor, list)
        assert isinstance(system_info.action_memory, MemoryStatsSchema)
        assert isinstance(system_info.observation_memory, MemoryStatsSchema)
        assert isinstance(system_info.environment_memory, MemoryStatsSchema)
