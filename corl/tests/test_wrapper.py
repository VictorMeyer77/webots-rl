"""
Unit tests for corl.api.wrapper.Wrapper

All HTTP calls are intercepted with pytest-mock / unittest.mock so no real
server is needed.
"""

from unittest.mock import MagicMock

import pytest
import requests

from corl.api.wrapper import Wrapper
from corl.schemas.api import Endpoint
from corl.schemas.learning import Action, Environment, Observation
from corl.schemas.tracker import StepKey
from corl.utils.config import Config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRAIN_ID = "train_abc"
WORKER_ID = 1
EPISODE_ID = 0
STEP = 5
BASE_URL = "http://localhost:8080/api/v1"


@pytest.fixture
def config() -> Config:
    cfg = MagicMock(spec=Config)
    cfg.__getitem__ = lambda self, k: "http://localhost" if k == "API_HOST" else "8080"
    return cfg


@pytest.fixture
def wrapper(config) -> Wrapper:
    return Wrapper(config, timeout=5)


@pytest.fixture
def step_key() -> StepKey:
    return StepKey(worker_id=WORKER_ID, episode_id=EPISODE_ID, step=STEP)


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()  # no-op by default
    return resp


# ===========================================================================
# Wrapper initialisation & context manager
# ===========================================================================


class TestWrapperInit:
    def test_base_url_is_built_from_config(self, wrapper):
        assert wrapper.base_url == "http://localhost:8080/api/v1"

    def test_timeout_is_set(self, wrapper):
        assert wrapper.timeout == 5

    def test_session_is_created(self, wrapper):
        assert wrapper.session is not None

    def test_context_manager_closes_session(self, config):
        with Wrapper(config) as w:
            session_mock = MagicMock()
            w.session = session_mock
        session_mock.close.assert_called_once()

    def test_close_closes_session(self, wrapper):
        session_mock = MagicMock()
        wrapper.session = session_mock
        wrapper.close()
        session_mock.close.assert_called_once()


# ===========================================================================
# get()
# ===========================================================================


class TestGet:
    def test_returns_dict_on_success(self, wrapper):
        wrapper.session.get = MagicMock(
            return_value=_mock_response({"action": 2, "executed": False})
        )
        result = wrapper.get(Endpoint.ACTION, TRAIN_ID, WORKER_ID, EPISODE_ID, STEP)
        assert result == {"action": 2, "executed": False}

    def test_url_is_correct(self, wrapper):
        wrapper.session.get = MagicMock(return_value=_mock_response({}))
        wrapper.get(Endpoint.ACTION, TRAIN_ID, WORKER_ID, EPISODE_ID, STEP)
        expected_url = f"{BASE_URL}/action/{TRAIN_ID}/{WORKER_ID}/{EPISODE_ID}/{STEP}"
        wrapper.session.get.assert_called_once_with(expected_url, timeout=5)

    def test_returns_none_on_http_error(self, wrapper):
        err_resp = MagicMock()
        err_resp.content = b"not found"
        wrapper.session.get = MagicMock(
            side_effect=requests.HTTPError(response=err_resp)
        )
        assert (
            wrapper.get(Endpoint.ACTION, TRAIN_ID, WORKER_ID, EPISODE_ID, STEP) is None
        )

    def test_returns_none_on_unexpected_error(self, wrapper):
        wrapper.session.get = MagicMock(side_effect=RuntimeError("unexpected"))
        assert (
            wrapper.get(Endpoint.ACTION, TRAIN_ID, WORKER_ID, EPISODE_ID, STEP) is None
        )

    def test_returns_none_when_raise_for_status_raises(self, wrapper):
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.HTTPError(
            response=MagicMock(content=b"err")
        )
        wrapper.session.get = MagicMock(return_value=resp)
        assert (
            wrapper.get(Endpoint.ACTION, TRAIN_ID, WORKER_ID, EPISODE_ID, STEP) is None
        )


# ===========================================================================
# post()
# ===========================================================================


class TestPost:
    def test_returns_true_on_success(self, wrapper):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"status": "success"})
        )
        assert (
            wrapper.post(Endpoint.ACTION, TRAIN_ID, WORKER_ID, EPISODE_ID, STEP, {})
            is True
        )

    def test_returns_false_on_non_success_status(self, wrapper):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"status": "error"})
        )
        assert (
            wrapper.post(Endpoint.ACTION, TRAIN_ID, WORKER_ID, EPISODE_ID, STEP, {})
            is False
        )

    def test_url_is_correct(self, wrapper):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"status": "success"})
        )
        wrapper.post(
            Endpoint.OBSERVATION, TRAIN_ID, WORKER_ID, EPISODE_ID, STEP, {"k": "v"}
        )
        expected_url = (
            f"{BASE_URL}/observation/{TRAIN_ID}/{WORKER_ID}/{EPISODE_ID}/{STEP}"
        )
        wrapper.session.post.assert_called_once_with(
            expected_url, json={"k": "v"}, timeout=5
        )

    def test_returns_false_on_http_error(self, wrapper):
        err_resp = MagicMock()
        err_resp.content = b"bad request"
        wrapper.session.post = MagicMock(
            side_effect=requests.HTTPError(response=err_resp)
        )
        assert (
            wrapper.post(Endpoint.ACTION, TRAIN_ID, WORKER_ID, EPISODE_ID, STEP, {})
            is False
        )

    def test_returns_false_on_unexpected_error(self, wrapper):
        wrapper.session.post = MagicMock(side_effect=Exception("boom"))
        assert (
            wrapper.post(Endpoint.ACTION, TRAIN_ID, WORKER_ID, EPISODE_ID, STEP, {})
            is False
        )


# ===========================================================================
# _get_batch_payload / _post_batch_payload (static helpers)
# ===========================================================================


class TestBatchPayloadHelpers:
    def test_get_batch_payload_structure(self):
        keys = [StepKey(worker_id=1, episode_id=2, step=3)]
        payload = Wrapper._get_batch_payload("t1", keys)
        assert payload == {
            "keys": [{"train_id": "t1", "worker_id": 1, "episode_id": 2, "step": 3}]
        }

    def test_get_batch_payload_multiple_keys(self):
        keys = [
            StepKey(worker_id=0, episode_id=0, step=0),
            StepKey(worker_id=1, episode_id=2, step=3),
        ]
        payload = Wrapper._get_batch_payload("t1", keys)
        assert len(payload["keys"]) == 2

    def test_post_batch_payload_structure(self):
        key = StepKey(worker_id=1, episode_id=2, step=3)
        action = Action(action=4, executed=True)
        payload = Wrapper._post_batch_payload("t1", [(key, action)])
        assert len(payload["items"]) == 1
        item = payload["items"][0]
        assert item["key"] == {
            "train_id": "t1",
            "worker_id": 1,
            "episode_id": 2,
            "step": 3,
        }
        assert item["value"] == {"action": 4, "executed": True}

    def test_post_batch_payload_multiple_values(self):
        values = [
            (StepKey(worker_id=0, episode_id=0, step=i), Action(action=i))
            for i in range(3)
        ]
        payload = Wrapper._post_batch_payload("t1", values)
        assert len(payload["items"]) == 3


# ===========================================================================
# get_batch()
# ===========================================================================


class TestGetBatch:
    def test_returns_results_on_success(self, wrapper, step_key):
        resp_data = {
            "results": [{"key": step_key.model_dump(), "value": {"data": {}}}],
            "missing": 0,
        }
        wrapper.session.post = MagicMock(return_value=_mock_response(resp_data))
        result = wrapper.get_batch(Endpoint.OBSERVATION, TRAIN_ID, [step_key])
        assert len(result) == 1

    def test_returns_empty_list_on_http_error(self, wrapper, step_key):
        err_resp = MagicMock()
        err_resp.content = b"error"
        wrapper.session.post = MagicMock(
            side_effect=requests.HTTPError(response=err_resp)
        )
        assert wrapper.get_batch(Endpoint.OBSERVATION, TRAIN_ID, [step_key]) == []

    def test_returns_empty_list_on_unexpected_error(self, wrapper, step_key):
        wrapper.session.post = MagicMock(side_effect=RuntimeError("oops"))
        assert wrapper.get_batch(Endpoint.OBSERVATION, TRAIN_ID, [step_key]) == []

    def test_returns_empty_list_when_no_results_key(self, wrapper, step_key):
        wrapper.session.post = MagicMock(return_value=_mock_response({"missing": 0}))
        assert wrapper.get_batch(Endpoint.OBSERVATION, TRAIN_ID, [step_key]) == []

    def test_url_is_correct(self, wrapper, step_key):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"results": [], "missing": 0})
        )
        wrapper.get_batch(Endpoint.ACTION, TRAIN_ID, [step_key])
        expected_url = f"{BASE_URL}/action/batch"
        call_args = wrapper.session.post.call_args
        assert call_args[0][0] == expected_url


# ===========================================================================
# post_batch()
# ===========================================================================


class TestPostBatch:
    def test_returns_true_on_success(self, wrapper, step_key):
        action = Action(action=1)
        wrapper.session.post = MagicMock(return_value=_mock_response({"total": 1}))
        assert (
            wrapper.post_batch(Endpoint.ACTION, TRAIN_ID, [(step_key, action)]) is True
        )

    def test_returns_false_when_total_mismatch(self, wrapper, step_key):
        action = Action(action=1)
        wrapper.session.post = MagicMock(return_value=_mock_response({"total": 0}))
        assert (
            wrapper.post_batch(Endpoint.ACTION, TRAIN_ID, [(step_key, action)]) is False
        )

    def test_returns_false_on_http_error(self, wrapper, step_key):
        err_resp = MagicMock()
        err_resp.content = b"error"
        wrapper.session.post = MagicMock(
            side_effect=requests.HTTPError(response=err_resp)
        )
        assert (
            wrapper.post_batch(
                Endpoint.ACTION, TRAIN_ID, [(step_key, Action(action=0))]
            )
            is False
        )

    def test_returns_false_on_unexpected_error(self, wrapper, step_key):
        wrapper.session.post = MagicMock(side_effect=Exception("boom"))
        assert (
            wrapper.post_batch(
                Endpoint.ACTION, TRAIN_ID, [(step_key, Action(action=0))]
            )
            is False
        )

    def test_url_is_correct(self, wrapper, step_key):
        wrapper.session.post = MagicMock(return_value=_mock_response({"total": 1}))
        wrapper.post_batch(Endpoint.ACTION, TRAIN_ID, [(step_key, Action(action=0))])
        expected_url = f"{BASE_URL}/action/batch/publish"
        call_args = wrapper.session.post.call_args
        assert call_args[0][0] == expected_url


# ===========================================================================
# Supervisor: create_training_session()
# ===========================================================================


class TestCreateTrainingSession:
    def test_succeeds_on_success_status(self, wrapper):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"status": "success"})
        )
        wrapper.create_training_session(TRAIN_ID)  # should not raise

    def test_raises_runtime_error_on_non_success(self, wrapper):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"status": "error"})
        )
        with pytest.raises(RuntimeError):
            wrapper.create_training_session(TRAIN_ID)

    def test_propagates_http_error(self, wrapper):
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.HTTPError()
        wrapper.session.post = MagicMock(return_value=resp)
        with pytest.raises(requests.HTTPError):
            wrapper.create_training_session(TRAIN_ID)

    def test_url_and_payload(self, wrapper):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"status": "success"})
        )
        wrapper.create_training_session(TRAIN_ID)
        wrapper.session.post.assert_called_once_with(
            f"{BASE_URL}/supervisor/train",
            json={"train_id": TRAIN_ID},
            timeout=5,
        )


# ===========================================================================
# Supervisor: add_worker()
# ===========================================================================


class TestAddWorker:
    def test_returns_worker_id(self, wrapper):
        wrapper.session.post = MagicMock(return_value=_mock_response({"worker_id": 42}))
        assert wrapper.add_worker(TRAIN_ID) == 42

    def test_raises_value_error_when_worker_id_missing(self, wrapper):
        wrapper.session.post = MagicMock(return_value=_mock_response({}))
        with pytest.raises(ValueError):
            wrapper.add_worker(TRAIN_ID)

    def test_propagates_http_error(self, wrapper):
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.HTTPError()
        wrapper.session.post = MagicMock(return_value=resp)
        with pytest.raises(requests.HTTPError):
            wrapper.add_worker(TRAIN_ID)

    def test_url_is_correct(self, wrapper):
        wrapper.session.post = MagicMock(return_value=_mock_response({"worker_id": 1}))
        wrapper.add_worker(TRAIN_ID)
        wrapper.session.post.assert_called_once_with(
            f"{BASE_URL}/supervisor/train/{TRAIN_ID}/worker", timeout=5
        )


# ===========================================================================
# Supervisor: get_workers()
# ===========================================================================


class TestGetWorkers:
    def test_returns_worker_list(self, wrapper):
        workers = [{"worker_id": 0, "active": True}, {"worker_id": 1, "active": False}]
        wrapper.session.get = MagicMock(
            return_value=_mock_response({"workers": workers})
        )
        assert wrapper.get_workers(TRAIN_ID) == workers

    def test_returns_empty_list_when_no_workers_key(self, wrapper):
        wrapper.session.get = MagicMock(return_value=_mock_response({}))
        assert wrapper.get_workers(TRAIN_ID) == []

    def test_propagates_http_error(self, wrapper):
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.HTTPError()
        wrapper.session.get = MagicMock(return_value=resp)
        with pytest.raises(requests.HTTPError):
            wrapper.get_workers(TRAIN_ID)


# ===========================================================================
# Supervisor: get_episode_id()
# ===========================================================================


class TestGetEpisodeId:
    def test_returns_episode_id(self, wrapper):
        wrapper.session.get = MagicMock(return_value=_mock_response({"episode_id": 7}))
        assert wrapper.get_episode_id(TRAIN_ID, WORKER_ID) == 7

    def test_raises_value_error_when_episode_id_missing(self, wrapper):
        wrapper.session.get = MagicMock(return_value=_mock_response({}))
        with pytest.raises(ValueError):
            wrapper.get_episode_id(TRAIN_ID, WORKER_ID)

    def test_episode_id_is_cast_to_int(self, wrapper):
        wrapper.session.get = MagicMock(
            return_value=_mock_response({"episode_id": "3"})
        )
        assert wrapper.get_episode_id(TRAIN_ID, WORKER_ID) == 3
        assert isinstance(wrapper.get_episode_id(TRAIN_ID, WORKER_ID), int)

    def test_propagates_http_error(self, wrapper):
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.HTTPError()
        wrapper.session.get = MagicMock(return_value=resp)
        with pytest.raises(requests.HTTPError):
            wrapper.get_episode_id(TRAIN_ID, WORKER_ID)


# ===========================================================================
# Supervisor: increment_episode_id()
# ===========================================================================


class TestIncrementEpisodeId:
    def test_succeeds_silently(self, wrapper):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"status": "success"})
        )
        wrapper.increment_episode_id(TRAIN_ID, WORKER_ID)  # no exception

    def test_raises_runtime_error_on_non_success(self, wrapper):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"status": "fail"})
        )
        with pytest.raises(RuntimeError):
            wrapper.increment_episode_id(TRAIN_ID, WORKER_ID)

    def test_propagates_http_error(self, wrapper):
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.HTTPError()
        wrapper.session.post = MagicMock(return_value=resp)
        with pytest.raises(requests.HTTPError):
            wrapper.increment_episode_id(TRAIN_ID, WORKER_ID)

    def test_url_is_correct(self, wrapper):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"status": "success"})
        )
        wrapper.increment_episode_id(TRAIN_ID, WORKER_ID)
        expected = f"{BASE_URL}/supervisor/train/{TRAIN_ID}/worker/{WORKER_ID}/episode/increment"
        wrapper.session.post.assert_called_once_with(expected, timeout=5)


# ===========================================================================
# Supervisor: update_worker_status()
# ===========================================================================


class TestUpdateWorkerStatus:
    def test_succeeds_silently(self, wrapper):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"status": "success"})
        )
        wrapper.update_worker_status(TRAIN_ID, WORKER_ID, True)  # no exception

    def test_raises_runtime_error_on_non_success(self, wrapper):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"status": "fail"})
        )
        with pytest.raises(RuntimeError):
            wrapper.update_worker_status(TRAIN_ID, WORKER_ID, True)

    def test_propagates_http_error(self, wrapper):
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.HTTPError()
        wrapper.session.post = MagicMock(return_value=resp)
        with pytest.raises(requests.HTTPError):
            wrapper.update_worker_status(TRAIN_ID, WORKER_ID, True)

    def test_payload_contains_status(self, wrapper):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"status": "success"})
        )
        wrapper.update_worker_status(TRAIN_ID, WORKER_ID, False)
        call_kwargs = wrapper.session.post.call_args[1]
        assert call_kwargs["json"] == {"worker_status": False}


# ===========================================================================
# Supervisor: get_worker_status()
# ===========================================================================


class TestGetWorkerStatus:
    def test_returns_true_when_active(self, wrapper):
        wrapper.session.get = MagicMock(return_value=_mock_response({"status": True}))
        assert wrapper.get_worker_status(TRAIN_ID, WORKER_ID) is True

    def test_returns_false_when_inactive(self, wrapper):
        wrapper.session.get = MagicMock(return_value=_mock_response({"status": False}))
        assert wrapper.get_worker_status(TRAIN_ID, WORKER_ID) is False

    def test_raises_value_error_when_status_missing(self, wrapper):
        wrapper.session.get = MagicMock(return_value=_mock_response({}))
        with pytest.raises(ValueError):
            wrapper.get_worker_status(TRAIN_ID, WORKER_ID)

    def test_propagates_http_error(self, wrapper):
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.HTTPError()
        wrapper.session.get = MagicMock(return_value=resp)
        with pytest.raises(requests.HTTPError):
            wrapper.get_worker_status(TRAIN_ID, WORKER_ID)


# ===========================================================================
# Action endpoints
# ===========================================================================


class TestGetAction:
    def test_returns_action_on_success(self, wrapper):
        wrapper.session.get = MagicMock(
            return_value=_mock_response({"action": 3, "executed": False})
        )
        action = wrapper.get_action(TRAIN_ID, WORKER_ID, EPISODE_ID, STEP)
        assert isinstance(action, Action)
        assert action.action == 3
        assert action.executed is False

    def test_returns_none_on_failure(self, wrapper):
        wrapper.session.get = MagicMock(
            side_effect=requests.HTTPError(response=MagicMock(content=b"err"))
        )
        assert wrapper.get_action(TRAIN_ID, WORKER_ID, EPISODE_ID, STEP) is None

    def test_executed_true_is_preserved(self, wrapper):
        wrapper.session.get = MagicMock(
            return_value=_mock_response({"action": 0, "executed": True})
        )
        action = wrapper.get_action(TRAIN_ID, WORKER_ID, EPISODE_ID, STEP)
        assert action.executed is True


class TestSendAction:
    def test_returns_true_on_success(self, wrapper):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"status": "success"})
        )
        assert (
            wrapper.send_action(TRAIN_ID, WORKER_ID, EPISODE_ID, STEP, Action(action=1))
            is True
        )

    def test_returns_false_on_failure(self, wrapper):
        wrapper.session.post = MagicMock(side_effect=Exception("err"))
        assert (
            wrapper.send_action(TRAIN_ID, WORKER_ID, EPISODE_ID, STEP, Action(action=1))
            is False
        )

    def test_sends_model_dump(self, wrapper):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"status": "success"})
        )
        action = Action(action=2, executed=True)
        wrapper.send_action(TRAIN_ID, WORKER_ID, EPISODE_ID, STEP, action)
        call_kwargs = wrapper.session.post.call_args[1]
        assert call_kwargs["json"] == {"action": 2, "executed": True}


class TestSendActionBatch:
    def test_returns_true_on_success(self, wrapper, step_key):
        wrapper.session.post = MagicMock(return_value=_mock_response({"total": 1}))
        assert (
            wrapper.send_action_batch(TRAIN_ID, [(step_key, Action(action=0))]) is True
        )

    def test_returns_false_on_failure(self, wrapper, step_key):
        wrapper.session.post = MagicMock(side_effect=Exception("err"))
        assert (
            wrapper.send_action_batch(TRAIN_ID, [(step_key, Action(action=0))]) is False
        )


# ===========================================================================
# Observation endpoints
# ===========================================================================


class TestGetObservationBatch:
    def _build_obs_response(self, step_key: StepKey) -> dict:
        return {
            "results": [
                {
                    "key": step_key.model_dump(),
                    "value": {"data": {"sensor": 0.5}},
                }
            ],
            "missing": 0,
        }

    def test_returns_list_of_tuples(self, wrapper, step_key):
        wrapper.session.post = MagicMock(
            return_value=_mock_response(self._build_obs_response(step_key))
        )
        results = wrapper.get_observation_batch(TRAIN_ID, [step_key])
        assert len(results) == 1
        key, obs = results[0]
        assert isinstance(key, StepKey)
        assert isinstance(obs, Observation)
        assert obs.data == {"sensor": 0.5}

    def test_returns_empty_list_on_failure(self, wrapper, step_key):
        wrapper.session.post = MagicMock(side_effect=Exception("err"))
        assert wrapper.get_observation_batch(TRAIN_ID, [step_key]) == []

    def test_none_observation_when_value_is_none(self, wrapper, step_key):
        resp = {
            "results": [{"key": step_key.model_dump(), "value": None}],
            "missing": 0,
        }
        wrapper.session.post = MagicMock(return_value=_mock_response(resp))
        results = wrapper.get_observation_batch(TRAIN_ID, [step_key])
        key, obs = results[0]
        assert obs is None


class TestSendObservation:
    def test_returns_true_on_success(self, wrapper):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"status": "success"})
        )
        obs = Observation(data={"x": 1.0})
        assert (
            wrapper.send_observation(TRAIN_ID, WORKER_ID, EPISODE_ID, STEP, obs) is True
        )

    def test_returns_false_on_failure(self, wrapper):
        wrapper.session.post = MagicMock(side_effect=Exception("err"))
        assert (
            wrapper.send_observation(
                TRAIN_ID, WORKER_ID, EPISODE_ID, STEP, Observation()
            )
            is False
        )

    def test_sends_model_dump(self, wrapper):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"status": "success"})
        )
        obs = Observation(data={"lidar": [1, 2, 3]})
        wrapper.send_observation(TRAIN_ID, WORKER_ID, EPISODE_ID, STEP, obs)
        call_kwargs = wrapper.session.post.call_args[1]
        assert call_kwargs["json"] == {"data": {"lidar": [1, 2, 3]}}


# ===========================================================================
# Environment endpoints
# ===========================================================================


class TestSendEnvironment:
    def test_returns_true_on_success(self, wrapper):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"status": "success"})
        )
        env = Environment(done=True, reward=1.0)
        assert (
            wrapper.send_environment(TRAIN_ID, WORKER_ID, EPISODE_ID, STEP, env) is True
        )

    def test_returns_false_on_failure(self, wrapper):
        wrapper.session.post = MagicMock(side_effect=Exception("err"))
        assert (
            wrapper.send_environment(
                TRAIN_ID, WORKER_ID, EPISODE_ID, STEP, Environment()
            )
            is False
        )

    def test_sends_correct_payload(self, wrapper):
        wrapper.session.post = MagicMock(
            return_value=_mock_response({"status": "success"})
        )
        env = Environment(done=True, reward=-5.0, data={"info": "test"})
        wrapper.send_environment(TRAIN_ID, WORKER_ID, EPISODE_ID, STEP, env)
        call_kwargs = wrapper.session.post.call_args[1]
        assert call_kwargs["json"] == {
            "done": True,
            "reward": -5.0,
            "data": {"info": "test"},
        }


class TestGetEnvironmentBatch:
    def _build_env_response(self, step_key: StepKey) -> dict:
        return {
            "results": [
                {
                    "key": step_key.model_dump(),
                    "value": {"done": True, "reward": 2.5, "data": {}},
                }
            ],
            "missing": 0,
        }

    def test_returns_list_of_tuples(self, wrapper, step_key):
        wrapper.session.post = MagicMock(
            return_value=_mock_response(self._build_env_response(step_key))
        )
        results = wrapper.get_environment_batch(TRAIN_ID, [step_key])
        assert len(results) == 1
        key, env = results[0]
        assert isinstance(key, StepKey)
        assert isinstance(env, Environment)
        assert env.done is True
        assert env.reward == 2.5

    def test_returns_empty_list_on_failure(self, wrapper, step_key):
        wrapper.session.post = MagicMock(side_effect=Exception("err"))
        assert wrapper.get_environment_batch(TRAIN_ID, [step_key]) == []

    def test_none_environment_when_value_is_none(self, wrapper, step_key):
        resp = {
            "results": [{"key": step_key.model_dump(), "value": None}],
            "missing": 0,
        }
        wrapper.session.post = MagicMock(return_value=_mock_response(resp))
        results = wrapper.get_environment_batch(TRAIN_ID, [step_key])
        key, env = results[0]
        assert env is None

    def test_environment_data_field_defaults_to_empty_dict(self, wrapper, step_key):
        resp = {
            "results": [
                {"key": step_key.model_dump(), "value": {"done": False, "reward": 0.0}}
            ],
            "missing": 0,
        }
        wrapper.session.post = MagicMock(return_value=_mock_response(resp))
        results = wrapper.get_environment_batch(TRAIN_ID, [step_key])
        _, env = results[0]
        assert env.data == {}
