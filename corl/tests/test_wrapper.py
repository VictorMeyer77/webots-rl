import pytest
from unittest.mock import Mock, patch
import requests

from corl.api.wrapper import Wrapper
from corl.api.schemas import Action, Observation, Environment, Endpoint
from corl.utils.config import Config


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    config = Mock(spec=Config)
    config.get.side_effect = lambda key: {
        "API_HOST": "http://localhost",
        "API_PORT": "8000",
    }.get(key)
    return config


@pytest.fixture
def wrapper(mock_config):
    """Create a Wrapper instance with mock config."""
    return Wrapper(config=mock_config)


@pytest.fixture
def mock_session():
    """Create a mock session for testing."""
    with patch("requests.Session") as mock:
        session = Mock()
        mock.return_value = session
        yield session


class TestWrapperInitialization:
    """Test Wrapper initialization."""

    def test_init_with_default_config(self):
        """Test wrapper initialization with default config."""
        with patch("corl.utils.config.Config") as mock_config_class:
            mock_config_instance = Mock()
            mock_config_instance.get.side_effect = lambda key: {
                "API_HOST": "http://localhost",
                "API_PORT": "8000",
            }.get(key)
            mock_config_class.return_value = mock_config_instance

            wrapper = Wrapper()
            assert wrapper.base_url == "http://localhost:8000/api/v1"
            assert isinstance(wrapper.session, requests.Session)

    def test_init_with_custom_config(self, mock_config):
        """Test wrapper initialization with custom config."""
        wrapper = Wrapper(config=mock_config)
        assert wrapper.base_url == "http://localhost:8000/api/v1"
        assert isinstance(wrapper.session, requests.Session)


class TestGetMethod:
    """Test the generic get method."""

    def test_get_success(self, wrapper):
        """Test successful GET request."""
        mock_response = Mock()
        mock_response.json.return_value = {"test": "data"}
        wrapper.session.get = Mock(return_value=mock_response)

        result = wrapper.get(Endpoint.ACTION, "train1", 0, 1, 5)

        assert result == {"test": "data"}
        wrapper.session.get.assert_called_once_with(
            "http://localhost:8000/api/v1/action/train1/0/1/5",
            timeout=10,
        )

    def test_get_request_exception(self, wrapper):
        """Test GET request with RequestException."""
        mock_response = Mock()
        mock_response.content = b"Error message"
        exception = requests.RequestException()
        exception.response = mock_response
        wrapper.session.get = Mock(side_effect=exception)

        result = wrapper.get(Endpoint.ACTION, "train1", 0, 1, 5)

        assert result is None

    def test_get_unexpected_exception(self, wrapper):
        """Test GET request with unexpected exception."""
        wrapper.session.get = Mock(side_effect=ValueError("Unexpected error"))

        result = wrapper.get(Endpoint.ACTION, "train1", 0, 1, 5)

        assert result is None


class TestPostMethod:
    """Test the generic post method."""

    def test_post_success(self, wrapper):
        """Test successful POST request."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        wrapper.session.post = Mock(return_value=mock_response)

        data = {"test": "data"}
        result = wrapper.post(Endpoint.ACTION, "train1", 0, 1, 5, data)

        assert result is True
        wrapper.session.post.assert_called_once_with(
            "http://localhost:8000/api/v1/action/train1/0/1/5", json=data, timeout=10
        )

    def test_post_unexpected_response(self, wrapper):
        """Test POST request with unexpected response."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "error"}
        wrapper.session.post = Mock(return_value=mock_response)

        result = wrapper.post(Endpoint.ACTION, "train1", 0, 1, 5, {})

        assert result is False

    def test_post_request_exception(self, wrapper):
        """Test POST request with RequestException."""
        mock_response = Mock()
        mock_response.content = b"Error message"
        exception = requests.RequestException()
        exception.response = mock_response
        wrapper.session.post = Mock(side_effect=exception)

        result = wrapper.post(Endpoint.ACTION, "train1", 0, 1, 5, {})

        assert result is False

    def test_post_unexpected_exception(self, wrapper):
        """Test POST request with unexpected exception."""
        wrapper.session.post = Mock(side_effect=ValueError("Unexpected error"))

        result = wrapper.post(Endpoint.ACTION, "train1", 0, 1, 5, {})

        assert result is False


class TestSupervisorEndpoints:
    """Test supervisor-related endpoints."""

    def test_create_training_session_success(self, wrapper):
        """Test creating a training session successfully."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        wrapper.session.post = Mock(return_value=mock_response)

        wrapper.create_training_session("train1")

        wrapper.session.post.assert_called_once_with(
            "http://localhost:8000/api/v1/supervisor/train",
            json={"train_id": "train1"},
        )

    def test_create_training_session_failure(self, wrapper):
        """Test creating a training session with failure response."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "error"}
        wrapper.session.post = Mock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="Failed to create training session"):
            wrapper.create_training_session("train1")

    def test_add_worker_success(self, wrapper):
        """Test adding a worker successfully."""
        mock_response = Mock()
        mock_response.json.return_value = {"worker_id": 5}
        wrapper.session.post = Mock(return_value=mock_response)

        worker_id = wrapper.add_worker("train1")

        assert worker_id == 5
        wrapper.session.post.assert_called_once_with(
            "http://localhost:8000/api/v1/supervisor/train/train1/worker"
        )

    def test_get_workers_success(self, wrapper):
        """Test getting workers list successfully."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "workers": [
                {"worker_id": 0, "active": True},
                {"worker_id": 1, "active": False},
            ]
        }
        wrapper.session.get = Mock(return_value=mock_response)

        workers = wrapper.get_workers("train1")

        assert len(workers) == 2
        assert workers[0]["worker_id"] == 0
        wrapper.session.get.assert_called_once_with(
            "http://localhost:8000/api/v1/supervisor/train/train1"
        )

    def test_get_workers_empty(self, wrapper):
        """Test getting workers when none exist."""
        mock_response = Mock()
        mock_response.json.return_value = {}
        wrapper.session.get = Mock(return_value=mock_response)

        workers = wrapper.get_workers("train1")

        assert workers == []

    def test_get_episode_id_success(self, wrapper):
        """Test getting episode ID successfully."""
        mock_response = Mock()
        mock_response.json.return_value = {"episode_id": 42}
        wrapper.session.get = Mock(return_value=mock_response)

        episode_id = wrapper.get_episode_id("train1", 0)

        assert episode_id == 42
        wrapper.session.get.assert_called_once_with(
            "http://localhost:8000/api/v1/supervisor/train/train1/worker/0/episode"
        )

    def test_increment_episode_id_success(self, wrapper):
        """Test incrementing episode ID successfully."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        wrapper.session.post = Mock(return_value=mock_response)

        wrapper.increment_episode_id("train1", 0)

        wrapper.session.post.assert_called_once_with(
            "http://localhost:8000/api/v1/supervisor/train/train1/worker/0/episode/increment"
        )

    def test_increment_episode_id_failure(self, wrapper):
        """Test incrementing episode ID with failure response."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "error"}
        wrapper.session.post = Mock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="Failed to increment episode ID"):
            wrapper.increment_episode_id("train1", 0)

    def test_update_worker_status_success(self, wrapper):
        """Test updating worker status successfully."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        wrapper.session.post = Mock(return_value=mock_response)

        wrapper.update_worker_status("train1", 0, True)

        wrapper.session.post.assert_called_once_with(
            "http://localhost:8000/api/v1/supervisor/train/train1/worker/0/status",
            json={"worker_status": True},
        )

    def test_update_worker_status_failure(self, wrapper):
        """Test updating worker status with failure response."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "error"}
        wrapper.session.post = Mock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="Failed to update worker status"):
            wrapper.update_worker_status("train1", 0, False)


class TestActionEndpoints:
    """Test action-related endpoints."""

    def test_get_action_success(self, wrapper):
        """Test getting an action successfully."""
        wrapper.get = Mock(return_value={"action": 3, "executed": True})

        action = wrapper.get_action("train1", 0, 1, 5)

        assert isinstance(action, Action)
        assert action.action == 3
        assert action.executed is True
        wrapper.get.assert_called_once_with(Endpoint.ACTION, "train1", 0, 1, 5)

    def test_get_action_not_found(self, wrapper):
        """Test getting an action that doesn't exist."""
        wrapper.get = Mock(return_value=None)

        action = wrapper.get_action("train1", 0, 1, 5)

        assert action is None

    def test_send_action_success(self, wrapper):
        """Test sending an action successfully."""
        wrapper.post = Mock(return_value=True)
        action = Action(action=3, executed=False)

        result = wrapper.send_action("train1", 0, 1, 5, action)

        assert result is True
        wrapper.post.assert_called_once_with(
            Endpoint.ACTION,
            "train1",
            0,
            1,
            5,
            {"action": 3, "executed": False},
        )

    def test_send_action_failure(self, wrapper):
        """Test sending an action with failure."""
        wrapper.post = Mock(return_value=False)
        action = Action(action=3, executed=False)

        result = wrapper.send_action("train1", 0, 1, 5, action)

        assert result is False


class TestObservationEndpoints:
    """Test observation-related endpoints."""

    def test_get_observation_success(self, wrapper):
        """Test getting an observation successfully."""
        wrapper.get = Mock(return_value={"data": {"image": [1, 2, 3], "sensor": 0.5}})

        observation = wrapper.get_observation("train1", 0, 1, 5)

        assert isinstance(observation, Observation)
        assert observation.data == {"image": [1, 2, 3], "sensor": 0.5}
        wrapper.get.assert_called_once_with(Endpoint.OBSERVATION, "train1", 0, 1, 5)

    def test_get_observation_not_found(self, wrapper):
        """Test getting an observation that doesn't exist."""
        wrapper.get = Mock(return_value=None)

        observation = wrapper.get_observation("train1", 0, 1, 5)

        assert observation is None

    def test_send_observation_success(self, wrapper):
        """Test sending an observation successfully."""
        wrapper.post = Mock(return_value=True)
        observation = Observation(data={"image": [1, 2, 3]})

        result = wrapper.send_observation("train1", 0, 1, 5, observation)

        assert result is True
        wrapper.post.assert_called_once_with(
            Endpoint.OBSERVATION,
            "train1",
            0,
            1,
            5,
            {"data": {"image": [1, 2, 3]}},
        )

    def test_send_observation_failure(self, wrapper):
        """Test sending an observation with failure."""
        wrapper.post = Mock(return_value=False)
        observation = Observation(data={"image": [1, 2, 3]})

        result = wrapper.send_observation("train1", 0, 1, 5, observation)

        assert result is False


class TestEnvironmentEndpoints:
    """Test environment-related endpoints."""

    def test_send_environment_success(self, wrapper):
        """Test sending environment state successfully."""
        wrapper.post = Mock(return_value=True)
        env = Environment(done=False, reward=1.5, data={"info": "test"})

        result = wrapper.send_environment("train1", 0, 1, 5, env)

        assert result is True
        wrapper.post.assert_called_once_with(
            Endpoint.ENVIRONMENT,
            "train1",
            0,
            1,
            5,
            {"done": False, "reward": 1.5, "data": {"info": "test"}},
        )

    def test_send_environment_failure(self, wrapper):
        """Test sending environment state with failure."""
        wrapper.post = Mock(return_value=False)
        env = Environment(done=True, reward=0.0)

        result = wrapper.send_environment("train1", 0, 1, 5, env)

        assert result is False

    def test_get_environment_success(self, wrapper):
        """Test getting environment state successfully."""
        wrapper.get = Mock(
            return_value={"reward": 2.5, "done": False, "data": {"info": "test"}}
        )

        env = wrapper.get_environment("train1", 0, 1, 5)

        assert isinstance(env, Environment)
        assert env.reward == 2.5
        assert env.done is False
        assert env.data == {"info": "test"}
        wrapper.get.assert_called_once_with(Endpoint.ENVIRONMENT, "train1", 0, 1, 5)

    def test_get_environment_without_data(self, wrapper):
        """Test getting environment state without optional data field."""
        wrapper.get = Mock(return_value={"reward": 1.0, "done": True})

        env = wrapper.get_environment("train1", 0, 1, 5)

        assert isinstance(env, Environment)
        assert env.reward == 1.0
        assert env.done is True
        assert env.data == {}

    def test_get_environment_not_found(self, wrapper):
        """Test getting environment state that doesn't exist."""
        wrapper.get = Mock(return_value=None)

        env = wrapper.get_environment("train1", 0, 1, 5)

        assert env is None


class TestEndpointURLConstruction:
    """Test URL construction for different endpoints."""

    def test_url_with_leading_slash(self, wrapper):
        """Test that URLs are constructed correctly even with leading slash in endpoint."""
        mock_response = Mock()
        mock_response.json.return_value = {"test": "data"}
        wrapper.session.get = Mock(return_value=mock_response)

        # Using Endpoint enum which doesn't have leading slash
        wrapper.get(Endpoint.ACTION, "train1", 2, 3, 10)

        # Verify URL doesn't have double slashes
        called_url = wrapper.session.get.call_args[0][0]
        assert "//" not in called_url.replace("http://", "")
        assert called_url == "http://localhost:8000/api/v1/action/train1/2/3/10"
