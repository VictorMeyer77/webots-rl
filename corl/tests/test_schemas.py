"""
Unit tests for API schemas (Action, Observation, Environment, Endpoint).

Tests cover:
- Model instantiation with valid/invalid data
- Type validation and coercion
- Serialization (to_dict, model_dump, model_dump_json)
- Deserialization (model_validate, model_validate_json)
- Default values and field factories
- Edge cases (empty dicts, None values, type mismatches)
"""

import pytest
from pydantic import ValidationError

from corl.api.schemas import Action, Observation, Environment, Endpoint


class TestEnvironment:
    """Test Environment model validation and serialization."""

    def test_default_initialization(self):
        """Test Environment with default values."""
        env = Environment()
        assert env.done is False
        assert env.reward == 0.0
        assert env.data == {}

    def test_full_initialization(self):
        """Test Environment with all fields provided."""
        env = Environment(done=True, reward=10.5, data={"info": "test"})
        assert env.done is True
        assert env.reward == 10.5
        assert env.data == {"info": "test"}

    def test_partial_initialization(self):
        """Test Environment with partial fields."""
        env = Environment(reward=5.0)
        assert env.done is False
        assert env.reward == 5.0
        assert env.data == {}

    def test_type_coercion(self):
        """Test that Pydantic coerces compatible types."""
        # int to float coercion for reward
        env = Environment(reward=10)
        assert env.reward == 10.0
        assert isinstance(env.reward, float)

        # int to bool coercion
        env = Environment(done=1)
        assert env.done is True

    def test_invalid_types(self):
        """Test that invalid types raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Environment(done="not_a_bool")
        assert "done" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Environment(reward="not_a_number")
        assert "reward" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Environment(data="not_a_dict")
        assert "data" in str(exc_info.value)

    def test_to_dict(self):
        """Test to_dict() method for backwards compatibility."""
        env = Environment(done=True, reward=2.5, data={"key": "value"})
        result = env.to_dict()
        assert result == {"done": True, "reward": 2.5, "data": {"key": "value"}}
        assert isinstance(result, dict)

    def test_model_dump(self):
        """Test Pydantic model_dump() method."""
        env = Environment(done=True, reward=2.5, data={"key": "value"})
        result = env.model_dump()
        assert result == {"done": True, "reward": 2.5, "data": {"key": "value"}}

    def test_model_dump_json(self):
        """Test JSON serialization."""
        env = Environment(done=True, reward=2.5, data={"key": "value"})
        json_str = env.model_dump_json()
        assert '"done":true' in json_str
        assert '"reward":2.5' in json_str
        assert '"key":"value"' in json_str

    def test_model_validate(self):
        """Test creating Environment from dict."""
        data = {"done": True, "reward": 3.14, "data": {"test": 123}}
        env = Environment.model_validate(data)
        assert env.done is True
        assert env.reward == 3.14
        assert env.data == {"test": 123}

    def test_model_validate_json(self):
        """Test creating Environment from JSON string."""
        json_str = '{"done": false, "reward": 1.5, "data": {"key": "val"}}'
        env = Environment.model_validate_json(json_str)
        assert env.done is False
        assert env.reward == 1.5
        assert env.data == {"key": "val"}

    def test_mutable_default_isolation(self):
        """Test that default dict is not shared between instances."""
        env1 = Environment()
        env1.data["key1"] = "value1"

        env2 = Environment()
        assert "key1" not in env2.data
        assert env1.data != env2.data

    def test_nested_data(self):
        """Test Environment with complex nested data."""
        env = Environment(
            done=False,
            reward=5.0,
            data={
                "sensors": [1, 2, 3],
                "position": {"x": 10, "y": 20},
                "metadata": {"step": 100, "episode": 5},
            },
        )
        assert env.data["sensors"] == [1, 2, 3]
        assert env.data["position"]["x"] == 10
        assert env.data["metadata"]["step"] == 100


class TestAction:
    """Test Action model validation and serialization."""

    def test_initialization(self):
        """Test Action initialization."""
        action = Action(action=5)
        assert action.action == 5
        assert action.executed is False

    def test_full_initialization(self):
        """Test Action with all fields."""
        action = Action(action=3, executed=True)
        assert action.action == 3
        assert action.executed is True

    def test_invalid_action_type(self):
        """Test that non-int action raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Action(action="not_an_int")
        assert "action" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Action(action=3.5)
        assert "action" in str(exc_info.value)

    def test_invalid_executed_type(self):
        """Test that non-bool executed raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Action(action=1, executed="not_bool")
        assert "executed" in str(exc_info.value)

    def test_to_dict(self):
        """Test to_dict() method."""
        action = Action(action=7, executed=True)
        result = action.to_dict()
        assert result == {"action": 7, "executed": True}

    def test_model_dump(self):
        """Test Pydantic model_dump()."""
        action = Action(action=7, executed=True)
        result = action.model_dump()
        assert result == {"action": 7, "executed": True}

    def test_model_validate(self):
        """Test creating Action from dict."""
        data = {"action": 10, "executed": False}
        action = Action.model_validate(data)
        assert action.action == 10
        assert action.executed is False

    def test_model_validate_json(self):
        """Test creating Action from JSON string."""
        json_str = '{"action": 5, "executed": true}'
        action = Action.model_validate_json(json_str)
        assert action.action == 5
        assert action.executed is True

    def test_negative_action(self):
        """Test Action with negative value."""
        action = Action(action=-1)
        assert action.action == -1

    def test_zero_action(self):
        """Test Action with zero value."""
        action = Action(action=0)
        assert action.action == 0


class TestObservation:
    """Test Observation model validation and serialization."""

    def test_default_initialization(self):
        """Test Observation with default empty dict."""
        obs = Observation()
        assert obs.data == {}

    def test_initialization_with_data(self):
        """Test Observation with data."""
        obs = Observation(data={"sensor1": 10, "sensor2": 20})
        assert obs.data == {"sensor1": 10, "sensor2": 20}

    def test_invalid_data_type(self):
        """Test that non-dict data raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Observation(data="not_a_dict")
        assert "data" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Observation(data=[1, 2, 3])
        assert "data" in str(exc_info.value)

    def test_to_dict(self):
        """Test to_dict() method."""
        obs = Observation(data={"key": "value"})
        result = obs.to_dict()
        assert result == {"data": {"key": "value"}}

    def test_model_dump(self):
        """Test Pydantic model_dump()."""
        obs = Observation(data={"key": "value"})
        result = obs.model_dump()
        assert result == {"data": {"key": "value"}}

    def test_model_validate(self):
        """Test creating Observation from dict."""
        data = {"data": {"sensors": [1, 2, 3], "image": "base64..."}}
        obs = Observation.model_validate(data)
        assert obs.data["sensors"] == [1, 2, 3]
        assert obs.data["image"] == "base64..."

    def test_model_validate_json(self):
        """Test creating Observation from JSON string."""
        json_str = '{"data": {"x": 1, "y": 2}}'
        obs = Observation.model_validate_json(json_str)
        assert obs.data == {"x": 1, "y": 2}

    def test_mutable_default_isolation(self):
        """Test that default dict is not shared between instances."""
        obs1 = Observation()
        obs1.data["key1"] = "value1"

        obs2 = Observation()
        assert "key1" not in obs2.data
        assert obs1.data != obs2.data

    def test_complex_observation_data(self):
        """Test Observation with complex nested data."""
        obs = Observation(
            data={
                "camera": {"width": 640, "height": 480, "pixels": [0] * 100},
                "lidar": {"ranges": [1.0, 2.0, 3.0], "angles": [0, 45, 90]},
                "gps": {"lat": 37.7749, "lon": -122.4194},
            }
        )
        assert obs.data["camera"]["width"] == 640
        assert len(obs.data["camera"]["pixels"]) == 100
        assert obs.data["lidar"]["ranges"] == [1.0, 2.0, 3.0]
        assert obs.data["gps"]["lat"] == 37.7749


class TestEndpoint:
    """Test Endpoint enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert Endpoint.ACTION == "action"
        assert Endpoint.OBSERVATION == "observation"
        assert Endpoint.ENVIRONMENT == "environment"
        assert Endpoint.SUPERVISOR == "supervisor"

    def test_enum_membership(self):
        """Test enum membership checks."""
        assert "action" in Endpoint._value2member_map_
        assert "observation" in Endpoint._value2member_map_
        assert "environment" in Endpoint._value2member_map_
        assert "supervisor" in Endpoint._value2member_map_

    def test_string_comparison(self):
        """Test that Endpoint can be compared to strings."""
        assert Endpoint.ACTION == "action"
        assert Endpoint.OBSERVATION == "observation"

    def test_string_formatting(self):
        """Test that Endpoint can be used in f-strings."""
        url = f"http://localhost:8000/{Endpoint.ACTION}/train1/0/1/5"
        assert url == "http://localhost:8000/action/train1/0/1/5"

    def test_iteration(self):
        """Test iterating over all endpoints."""
        endpoints = list(Endpoint)
        assert len(endpoints) == 4
        assert Endpoint.ACTION in endpoints
        assert Endpoint.OBSERVATION in endpoints
        assert Endpoint.ENVIRONMENT in endpoints
        assert Endpoint.SUPERVISOR in endpoints


class TestIntegration:
    """Test integration scenarios between models."""

    def test_round_trip_serialization(self):
        """Test serializing and deserializing all models."""
        # Environment
        env = Environment(done=True, reward=5.5, data={"info": "test"})
        env_dict = env.to_dict()
        env_restored = Environment.model_validate(env_dict)
        assert env_restored.done == env.done
        assert env_restored.reward == env.reward
        assert env_restored.data == env.data

        # Action
        action = Action(action=3, executed=True)
        action_dict = action.to_dict()
        action_restored = Action.model_validate(action_dict)
        assert action_restored.action == action.action
        assert action_restored.executed == action.executed

        # Observation
        obs = Observation(data={"sensor": 100})
        obs_dict = obs.to_dict()
        obs_restored = Observation.model_validate(obs_dict)
        assert obs_restored.data == obs.data

    def test_json_round_trip(self):
        """Test JSON serialization round trip."""
        env = Environment(done=True, reward=3.14, data={"key": "val"})
        json_str = env.model_dump_json()
        env_restored = Environment.model_validate_json(json_str)
        assert env_restored.done == env.done
        assert env_restored.reward == env.reward
        assert env_restored.data == env.data

    def test_api_workflow_simulation(self):
        """Simulate a typical API workflow."""
        # Agent creates action
        action = Action(action=2, executed=False)
        action_payload = action.to_dict()

        # API stores and retrieves action
        action_from_api = Action.model_validate(action_payload)
        action_from_api.executed = True

        # Environment creates observation
        obs = Observation(data={"position": [1, 2, 3], "velocity": [0.1, 0.2, 0.3]})
        obs_payload = obs.to_dict()

        # Agent receives observation
        obs_received = Observation.model_validate(obs_payload)
        assert obs_received.data["position"] == [1, 2, 3]

        # Environment creates state
        env_state = Environment(done=False, reward=1.0, data={"step": 10})
        env_payload = env_state.to_dict()

        # Trainer receives environment state
        env_received = Environment.model_validate(env_payload)
        assert env_received.reward == 1.0
        assert env_received.done is False
