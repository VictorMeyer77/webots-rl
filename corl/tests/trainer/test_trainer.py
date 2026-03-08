"""
Unit tests for corl.trainer.trainer.Trainer

Trainer is abstract so every test uses a minimal concrete subclass.
All external collaborators (Wrapper, Tracker) are mocked
so no real API server, GPU, or file system is required.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from corl.schemas.learning import Observation
from corl.schemas.tracker import StepKey
from corl.trainer.tracker import StepResult
from corl.utils.config import Config

# ---------------------------------------------------------------------------
# Minimal concrete subclass
# ---------------------------------------------------------------------------


def _make_trainer_class():
    import numpy as np
    from numpy.typing import NDArray

    from corl.trainer.trainer import Trainer

    class ConcreteTrainer(Trainer):
        def params(self) -> dict[str, str | int | float]:
            pass

        def run(self, epochs: int) -> None:
            pass

        def policy(self, observations: NDArray[np.float32]) -> NDArray[np.int32]:
            # Always returns action 0 for every observation
            return np.zeros(observations.shape[0], dtype=np.int32)

        def parse_observations(self, observations):
            return [
                (key, obs.data.get("array", np.zeros(3, dtype=np.float32)))
                for key, obs in observations
            ]

    return ConcreteTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRAIN_ID = "train_test_001"
EXPERIMENT_NAME = "test_experiment"
OUTPUT_DIR = ".train"


def _make_config() -> MagicMock:
    cfg = MagicMock(spec=Config)

    values = {
        "train_id": TRAIN_ID,
        "trainer_output_dir": OUTPUT_DIR,
        "trainer_worker_timeout": 30,
        "world_name": EXPERIMENT_NAME,
        "api_host": "http://localhost",
        "api_port": 8000,
    }
    cfg.get = MagicMock(side_effect=lambda key: values[key.lower()])
    cfg.__getitem__ = MagicMock(side_effect=lambda key: values[key.lower()])
    return cfg


def _make_model() -> MagicMock:
    return MagicMock()


@pytest.fixture
def trainer():
    ConcreteTrainer = _make_trainer_class()
    config = _make_config()
    model = _make_model()

    with (
        patch("corl.trainer.trainer.Wrapper") as MockWrapper,
        patch("corl.trainer.trainer.Tracker") as MockTracker,
        patch("corl.trainer.trainer.mlflow"),
        patch("os.makedirs"),
    ):
        mock_api = MagicMock()
        MockWrapper.return_value = mock_api

        mock_tracker = MagicMock()
        MockTracker.return_value = mock_tracker

        t = ConcreteTrainer(model, config)
        t.api = mock_api
        t.tracker = mock_tracker
        return t


def _step_key(worker_id=0, episode_id=0, step=0) -> StepKey:
    return StepKey(worker_id=worker_id, episode_id=episode_id, step=step)


def _complete_result(done=False) -> StepResult:
    r = StepResult()
    r.observation = np.zeros(3, dtype=np.float32)
    r.action = 0
    r.reward = 1.0
    r.done = done
    return r


# ===========================================================================
# __init__
# ===========================================================================


class TestInit:
    def test_train_id_is_set(self, trainer):
        assert trainer.train_id == TRAIN_ID

    def test_model_dir_contains_train_id(self, trainer):
        assert TRAIN_ID in trainer.model_dir

    def test_api_is_set(self, trainer):
        assert trainer.api is not None

    def test_tracker_is_set(self, trainer):
        assert trainer.tracker is not None

    def test_create_training_session_called_with_train_id(self):
        ConcreteTrainer = _make_trainer_class()
        config = _make_config()

        with (
            patch("corl.trainer.trainer.Wrapper") as MockWrapper,
            patch("corl.trainer.trainer.Tracker"),
            patch("corl.trainer.trainer.mlflow"),
            patch("os.makedirs"),
        ):
            mock_api = MagicMock()
            MockWrapper.return_value = mock_api
            ConcreteTrainer(_make_model(), config)
            mock_api.create_training_session.assert_called_once_with(TRAIN_ID)


# ===========================================================================
# close
# ===========================================================================


class TestClose:
    def _close(self, trainer):
        with (
            patch.object(trainer, "_generate_video"),
            patch.object(trainer, "_close_mlflow"),
            patch.object(trainer, "_close_model"),
        ):
            trainer.close()

    def test_close_calls_tracker_close_workers(self, trainer):
        self._close(trainer)
        trainer.tracker.close_workers.assert_called_once()

    def test_close_calls_api_close(self, trainer):
        self._close(trainer)
        trainer.api.close.assert_called_once()

    def test_close_calls_close_mlflow(self, trainer):
        with (
            patch.object(trainer, "_generate_video"),
            patch.object(trainer, "_close_mlflow") as mock_mlflow,
            patch.object(trainer, "_close_model"),
        ):
            trainer.close()
        mock_mlflow.assert_called_once()


# ===========================================================================
# _training_step_observation
# ===========================================================================


class TestTrainingStepObservation:
    def test_queries_none_observation_keys(self, trainer):
        key = _step_key()
        trainer.tracker.get_buffer_none_observations.return_value = [key]
        trainer.api.get_observation_batch.return_value = []
        trainer._training_step_observation()
        trainer.api.get_observation_batch.assert_called_once_with(TRAIN_ID, [key])

    def test_filters_out_none_observations(self, trainer):
        key = _step_key()
        trainer.tracker.get_buffer_none_observations.return_value = [key]
        trainer.api.get_observation_batch.return_value = [
            (key, None),
            (key, Observation(data={"array": np.zeros(3)})),
        ]
        _ = trainer._training_step_observation()
        # only the non-None observation should be parsed and added
        trainer.tracker.add_buffer_observations.assert_called_once()
        added = trainer.tracker.add_buffer_observations.call_args[0][0]
        assert len(added) == 1

    def test_returns_parsed_observation_arrays(self, trainer):
        key = _step_key()
        obs_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        trainer.tracker.get_buffer_none_observations.return_value = [key]
        trainer.api.get_observation_batch.return_value = [
            (key, Observation(data={"array": obs_array}))
        ]
        result = trainer._training_step_observation()
        assert len(result) == 1
        returned_key, returned_array = result[0]
        assert returned_key == key
        assert np.array_equal(returned_array, obs_array)

    def test_returns_empty_when_no_none_observations(self, trainer):
        trainer.tracker.get_buffer_none_observations.return_value = []
        trainer.api.get_observation_batch.return_value = []
        result = trainer._training_step_observation()
        assert result == []

    def test_adds_observations_to_tracker_buffer(self, trainer):
        key = _step_key()
        trainer.tracker.get_buffer_none_observations.return_value = [key]
        trainer.api.get_observation_batch.return_value = [
            (key, Observation(data={"array": np.zeros(3)}))
        ]
        trainer._training_step_observation()
        trainer.tracker.add_buffer_observations.assert_called_once()


# ===========================================================================
# _training_step_action
# ===========================================================================


class TestTrainingStepAction:
    def test_sends_action_batch_to_api(self, trainer):
        key = _step_key()
        obs = np.array([1.0, 2.0], dtype=np.float32)
        trainer.api.send_action_batch.return_value = True
        trainer._training_step_action([(key, obs)])
        trainer.api.send_action_batch.assert_called_once()

    def test_sends_correct_train_id(self, trainer):
        key = _step_key()
        obs = np.array([1.0], dtype=np.float32)
        trainer.api.send_action_batch.return_value = True
        trainer._training_step_action([(key, obs)])
        call_args = trainer.api.send_action_batch.call_args[0]
        assert call_args[0] == TRAIN_ID

    def test_adds_actions_to_tracker_buffer(self, trainer):
        key = _step_key()
        obs = np.array([1.0], dtype=np.float32)
        trainer.api.send_action_batch.return_value = True
        trainer._training_step_action([(key, obs)])
        trainer.tracker.add_buffer_actions.assert_called_once()

    def test_raises_runtime_error_when_send_fails(self, trainer):
        key = _step_key()
        obs = np.array([1.0], dtype=np.float32)
        trainer.api.send_action_batch.return_value = False
        with pytest.raises(RuntimeError, match="Failed to send action batch"):
            trainer._training_step_action([(key, obs)])

    def test_does_nothing_when_observations_empty(self, trainer):
        trainer._training_step_action([])
        trainer.api.send_action_batch.assert_not_called()
        trainer.tracker.add_buffer_actions.assert_not_called()

    def test_action_is_int_and_not_executed(self, trainer):
        key = _step_key()
        obs = np.array([1.0, 2.0], dtype=np.float32)
        trainer.api.send_action_batch.return_value = True
        trainer._training_step_action([(key, obs)])
        _, sent_pairs = trainer.api.send_action_batch.call_args[0]
        _, action = sent_pairs[0]
        assert isinstance(action.action, int)
        assert action.executed is False

    def test_multiple_observations_produce_multiple_actions(self, trainer):
        keys = [_step_key(worker_id=i) for i in range(3)]
        obs_list = [(k, np.zeros(2, dtype=np.float32)) for k in keys]
        trainer.api.send_action_batch.return_value = True
        trainer._training_step_action(obs_list)
        _, sent_pairs = trainer.api.send_action_batch.call_args[0]
        assert len(sent_pairs) == 3


# ===========================================================================
# _training_step_environment
# ===========================================================================


class TestTrainingStepEnvironment:
    def test_queries_none_environment_keys(self, trainer):
        key = _step_key()
        trainer.tracker.get_buffer_none_environments.return_value = [key]
        trainer.api.get_environment_batch.return_value = []
        trainer._training_step_environment()
        trainer.api.get_environment_batch.assert_called_once_with(TRAIN_ID, [key])

    def test_filters_out_none_environments(self, trainer):
        from corl.schemas.learning import Environment

        key0 = _step_key(worker_id=0)
        key1 = _step_key(worker_id=1)
        trainer.tracker.get_buffer_none_environments.return_value = [key0, key1]
        trainer.api.get_environment_batch.return_value = [
            (key0, None),
            (key1, Environment(reward=1.0, done=False)),
        ]
        trainer._training_step_environment()
        added = trainer.tracker.add_buffer_environments.call_args[0][0]
        assert len(added) == 1
        assert added[0][0] == key1

    def test_adds_environments_to_tracker(self, trainer):
        from corl.schemas.learning import Environment

        key = _step_key()
        trainer.tracker.get_buffer_none_environments.return_value = [key]
        trainer.api.get_environment_batch.return_value = [
            (key, Environment(reward=2.0, done=True))
        ]
        trainer._training_step_environment()
        trainer.tracker.add_buffer_environments.assert_called_once()

    def test_does_nothing_when_no_pending_environments(self, trainer):
        trainer.tracker.get_buffer_none_environments.return_value = []
        trainer.api.get_environment_batch.return_value = []
        trainer._training_step_environment()
        trainer.tracker.add_buffer_environments.assert_called_once_with([])


# ===========================================================================
# training_step
# ===========================================================================


class TestTrainingStep:
    def _setup_no_workers(self, trainer):
        trainer.tracker.worker_step_keys.return_value = []
        trainer.tracker.get_buffer_none_observations.return_value = []
        trainer.api.get_observation_batch.return_value = []
        trainer.tracker.get_buffer_none_environments.return_value = []
        trainer.api.get_environment_batch.return_value = []
        trainer.tracker.get_buffered_step_results.return_value = []

    def test_calls_tracker_refresh(self, trainer):
        self._setup_no_workers(trainer)
        with patch("corl.trainer.trainer.time.sleep"):
            trainer.training_step()
        trainer.tracker.refresh.assert_called_once()

    def test_returns_empty_when_no_workers(self, trainer):
        self._setup_no_workers(trainer)
        with patch("corl.trainer.trainer.time.sleep"):
            result = trainer.training_step()
        assert result == []

    def test_sleeps_when_no_workers(self, trainer):
        self._setup_no_workers(trainer)
        with patch("corl.trainer.trainer.time.sleep") as mock_sleep:
            trainer.training_step()
        mock_sleep.assert_called_once_with(1)

    def test_increments_episode_when_done(self, trainer):
        key = _step_key(worker_id=0, episode_id=1, step=3)
        result = _complete_result(done=True)

        trainer.tracker.worker_step_keys.return_value = [key]
        trainer.tracker.get_buffer_none_observations.return_value = [key]
        trainer.api.get_observation_batch.return_value = [
            (key, Observation(data={"array": np.zeros(3)}))
        ]
        trainer.api.send_action_batch.return_value = True
        trainer.tracker.get_buffer_none_environments.return_value = [key]
        trainer.api.get_environment_batch.return_value = []
        trainer.tracker.get_buffered_step_results.return_value = [(key, result)]

        trainer.training_step()
        trainer.tracker.increment_episode.assert_called_once_with(key.worker_id)
        trainer.tracker.increment_step.assert_not_called()

    def test_increments_step_when_not_done(self, trainer):
        key = _step_key(worker_id=0, episode_id=2, step=5)
        result = _complete_result(done=False)

        trainer.tracker.worker_step_keys.return_value = [key]
        trainer.tracker.get_buffer_none_observations.return_value = [key]
        trainer.api.get_observation_batch.return_value = [
            (key, Observation(data={"array": np.zeros(3)}))
        ]
        trainer.api.send_action_batch.return_value = True
        trainer.tracker.get_buffer_none_environments.return_value = [key]
        trainer.api.get_environment_batch.return_value = []
        trainer.tracker.get_buffered_step_results.return_value = [(key, result)]

        trainer.training_step()
        trainer.tracker.increment_step.assert_called_once_with(
            key.worker_id, key.episode_id
        )
        trainer.tracker.increment_episode.assert_not_called()

    def test_calls_worker_timeouts(self, trainer):
        key = _step_key()
        trainer.tracker.worker_step_keys.return_value = [key]
        trainer.tracker.get_buffer_none_observations.return_value = []
        trainer.api.get_observation_batch.return_value = []
        trainer.tracker.get_buffer_none_environments.return_value = []
        trainer.api.get_environment_batch.return_value = []
        trainer.tracker.get_buffered_step_results.return_value = []
        trainer.training_step()
        trainer.tracker.worker_timeouts.assert_called_once()

    def test_returns_step_results(self, trainer):
        key = _step_key()
        result = _complete_result(done=False)

        trainer.tracker.worker_step_keys.return_value = [key]
        trainer.tracker.get_buffer_none_observations.return_value = [key]
        trainer.api.get_observation_batch.return_value = [
            (key, Observation(data={"array": np.zeros(3)}))
        ]
        trainer.api.send_action_batch.return_value = True
        trainer.tracker.get_buffer_none_environments.return_value = [key]
        trainer.api.get_environment_batch.return_value = []
        trainer.tracker.get_buffered_step_results.return_value = [(key, result)]

        step_results = trainer.training_step()
        assert len(step_results) == 1
        assert step_results[0][0] == key
        assert step_results[0][1] is result

    def test_handles_multiple_done_and_not_done_workers(self, trainer):
        key_done = _step_key(worker_id=0, episode_id=1, step=2)
        key_not_done = _step_key(worker_id=1, episode_id=0, step=4)
        result_done = _complete_result(done=True)
        result_not_done = _complete_result(done=False)

        trainer.tracker.worker_step_keys.return_value = [key_done, key_not_done]
        trainer.tracker.get_buffer_none_observations.return_value = []
        trainer.api.get_observation_batch.return_value = []
        trainer.api.send_action_batch.return_value = True
        trainer.tracker.get_buffer_none_environments.return_value = []
        trainer.api.get_environment_batch.return_value = []
        trainer.tracker.get_buffered_step_results.return_value = [
            (key_done, result_done),
            (key_not_done, result_not_done),
        ]

        trainer.training_step()

        trainer.tracker.increment_episode.assert_called_once_with(key_done.worker_id)
        trainer.tracker.increment_step.assert_called_once_with(
            key_not_done.worker_id, key_not_done.episode_id
        )


# ===========================================================================
# Abstract interface
# ===========================================================================


class TestAbstractInterface:
    def test_cannot_instantiate_trainer_directly(self):
        from corl.trainer.trainer import Trainer

        with pytest.raises(TypeError):
            Trainer("train_id", _make_model(), "experiment", _make_config())  # noqa

    def test_save_model_is_abstract(self):
        from corl.trainer.trainer import Trainer

        assert "params" in Trainer.__abstractmethods__

    def test_run_is_abstract(self):
        from corl.trainer.trainer import Trainer

        assert "run" in Trainer.__abstractmethods__

    def test_policy_is_abstract(self):
        from corl.trainer.trainer import Trainer

        assert "policy" in Trainer.__abstractmethods__

    def test_parse_observations_is_abstract(self):
        from corl.trainer.trainer import Trainer

        assert "parse_observations" in Trainer.__abstractmethods__


# ===========================================================================
# _close_model
# ===========================================================================


class TestCloseModel:
    def _run(self, trainer, listdir_files, mlflow_mock=None):
        """Helper that patches os.listdir, mlflow, and shutil.rmtree."""
        with (
            patch("corl.trainer.trainer.os.listdir", return_value=listdir_files),
            patch("corl.trainer.trainer.mlflow") as mock_mlflow,
            patch("corl.trainer.trainer.shutil.rmtree") as mock_rmtree,
        ):
            trainer._close_model()
            return mock_mlflow, mock_rmtree

    def test_logs_non_checkpoint_files_to_mlflow(self, trainer):
        trainer.model_dir = "/model/dir"
        mock_mlflow, _ = self._run(trainer, ["weights.h5", "config.json"])
        assert mock_mlflow.log_artifact.call_count == 2
        paths = [c.args[0] for c in mock_mlflow.log_artifact.call_args_list]
        assert "/model/dir/weights.h5" in paths
        assert "/model/dir/config.json" in paths

    def test_skips_checkpoint_files(self, trainer):
        trainer.model_dir = "/model/dir"
        mock_mlflow, _ = self._run(
            trainer, ["weights.h5", "weights_ckt_001.h5", "_ckt_backup.h5"]
        )
        assert mock_mlflow.log_artifact.call_count == 1
        assert mock_mlflow.log_artifact.call_args.args[0] == "/model/dir/weights.h5"

    def test_all_checkpoint_files_skipped(self, trainer):
        trainer.model_dir = "/model/dir"
        mock_mlflow, _ = self._run(trainer, ["_ckt_1.h5", "epoch_ckt_2.h5"])
        mock_mlflow.log_artifact.assert_not_called()

    def test_empty_model_dir_does_not_log_or_raise(self, trainer):
        trainer.model_dir = "/model/dir"
        mock_mlflow, mock_rmtree = self._run(trainer, [])
        mock_mlflow.log_artifact.assert_not_called()
        mock_rmtree.assert_called_once_with("/model/dir")

    def test_artifact_path_is_model(self, trainer):
        trainer.model_dir = "/model/dir"
        mock_mlflow, _ = self._run(trainer, ["weights.h5"])
        assert mock_mlflow.log_artifact.call_args.kwargs["artifact_path"] == "model"

    def test_deletes_model_dir_after_upload(self, trainer):
        trainer.model_dir = "/model/dir"
        _, mock_rmtree = self._run(trainer, ["weights.h5"])
        mock_rmtree.assert_called_once_with("/model/dir")

    def test_deletes_dir_even_when_no_files_logged(self, trainer):
        trainer.model_dir = "/model/dir"
        _, mock_rmtree = self._run(trainer, ["_ckt_only.h5"])
        mock_rmtree.assert_called_once_with("/model/dir")


# ===========================================================================
# _generate_video
# ===========================================================================


class TestGenerateVideo:
    def _run(self, trainer, video_path=None, generate_raises=False):
        """Helper that patches generate_training_video, mlflow, and shutil.rmtree."""

        def _generate(path):
            if generate_raises:
                raise RuntimeError("ffmpeg not found")
            return video_path

        with (
            patch(
                "corl.trainer.trainer.generate_training_video",
                side_effect=_generate,
            ),
            patch("corl.trainer.trainer.mlflow") as mock_mlflow,
            patch("corl.trainer.trainer.shutil.rmtree") as mock_rmtree,
        ):
            trainer._generate_video()
            return mock_mlflow, mock_rmtree

    def test_logs_video_artifact_when_path_returned(self, trainer):
        trainer.video_dir = "/video/dir"
        mock_mlflow, _ = self._run(trainer, video_path="/video/dir/full.mp4")
        mock_mlflow.log_artifact.assert_called_once_with(
            "/video/dir/full.mp4", artifact_path="videos"
        )

    def test_does_not_log_artifact_when_no_video(self, trainer):
        trainer.video_dir = "/video/dir"
        mock_mlflow, _ = self._run(trainer, video_path=None)
        mock_mlflow.log_artifact.assert_not_called()

    def test_always_removes_video_dir(self, trainer):
        trainer.video_dir = "/video/dir"
        _, mock_rmtree = self._run(trainer, video_path="/video/dir/full.mp4")
        mock_rmtree.assert_called_once_with("/video/dir", ignore_errors=True)

    def test_removes_video_dir_even_when_no_video(self, trainer):
        trainer.video_dir = "/video/dir"
        _, mock_rmtree = self._run(trainer, video_path=None)
        mock_rmtree.assert_called_once_with("/video/dir", ignore_errors=True)

    def test_swallows_exception_from_generate(self, trainer):
        trainer.video_dir = "/video/dir"
        # Should not raise
        mock_mlflow, mock_rmtree = self._run(trainer, generate_raises=True)
        mock_mlflow.log_artifact.assert_not_called()

    def test_removes_video_dir_even_after_exception(self, trainer):
        trainer.video_dir = "/video/dir"
        _, mock_rmtree = self._run(trainer, generate_raises=True)
        mock_rmtree.assert_called_once_with("/video/dir", ignore_errors=True)

    def test_generate_training_video_called_with_video_dir(self, trainer):
        trainer.video_dir = "/video/dir"
        with (
            patch(
                "corl.trainer.trainer.generate_training_video", return_value=None
            ) as mock_gen,
            patch("corl.trainer.trainer.mlflow"),
            patch("corl.trainer.trainer.shutil.rmtree"),
        ):
            trainer._generate_video()
        mock_gen.assert_called_once_with("/video/dir")
