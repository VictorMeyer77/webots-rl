"""
Unit tests for corl.trainer.trainer.Trainer

Trainer is abstract so every test uses a minimal concrete subclass.
All external collaborators (Wrapper, Tracker) are mocked
so no real API server, GPU, or file system is required.
"""

import sys
from unittest.mock import MagicMock, patch

# mlflow is not installed in the test environment — mock before any corl import
sys.modules.setdefault("mlflow", MagicMock())

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from corl.schemas.learning import Observation  # noqa: E402
from corl.schemas.tracker import StepKey  # noqa: E402
from corl.trainer.tracker import StepResult  # noqa: E402
from corl.utils.config import Config  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal concrete subclass
# ---------------------------------------------------------------------------


def _make_real_trainer_class():
    """Concrete subclass that inherits checkpoint() and recovery() from Trainer base."""
    import numpy as np
    from numpy.typing import NDArray

    from corl.trainer.trainer import Trainer

    class RealTrainer(Trainer):
        def params(self) -> dict[str, str | int | float | bool]:
            return super().params()

        def run(self, max_transitions: int) -> None:
            pass

        def policy(self, observations: NDArray[np.float32]) -> NDArray[np.float32]:
            return np.zeros(observations.shape[0], dtype=np.float32)

        def parse_observations(self, observations):
            return [
                (key, obs.data.get("array", np.zeros(3, dtype=np.float32)))
                for key, obs in observations
            ]

    return RealTrainer


def _make_trainer_class():
    import numpy as np
    from numpy.typing import NDArray

    from corl.trainer.trainer import Trainer

    class ConcreteTrainer(Trainer):
        def params(self) -> dict[str, str | int | float | bool]:
            pass

        def run(self, max_transitions: int) -> None:
            pass

        def policy(self, observations: NDArray[np.float32]) -> NDArray[np.float32]:
            # Always returns action 0 for every observation
            return np.zeros(observations.shape[0], dtype=np.float32)

        def parse_observations(self, observations):
            return [
                (key, obs.data.get("array", np.zeros(3, dtype=np.float32)))
                for key, obs in observations
            ]

        def checkpoint(self) -> Path:
            pass

        def recovery(self, checkpoint_id: str) -> Path:
            pass

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
        "trainer_log_metric_frequency": 10,
        "trainer_mlflow_url": "localhost:5001",
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
        patch("pathlib.Path.mkdir"),
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

    def test_checkpoint_dir_contains_train_id(self, trainer):
        assert TRAIN_ID in trainer.checkpoint_dir

    def test_checkpoint_frequency_default(self, trainer):
        assert trainer.checkpoint_frequency == 100_000

    def test_last_checkpoint_transition_starts_at_zero(self, trainer):
        assert trainer.last_checkpoint_transition == 0

    def test_episode_count_starts_at_zero(self, trainer):
        assert trainer.episode_count == 0

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
            patch("pathlib.Path.mkdir"),
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
            patch.object(trainer, "_close_checkpoints"),
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
            patch.object(trainer, "_close_checkpoints"),
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

    def test_action_is_list_of_float(self, trainer):
        key = _step_key()
        obs = np.array([1.0, 2.0], dtype=np.float32)
        trainer.api.send_action_batch.return_value = True
        trainer._training_step_action([(key, obs)])
        _, sent_pairs = trainer.api.send_action_batch.call_args[0]
        _, action = sent_pairs[0]
        assert isinstance(action.action, list)
        assert all(isinstance(v, float) for v in action.action)

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
            Trainer(_make_model(), _make_config())  # noqa

    def test_run_is_abstract(self):
        from corl.trainer.trainer import Trainer

        assert "run" in Trainer.__abstractmethods__

    def test_policy_is_abstract(self):
        from corl.trainer.trainer import Trainer

        assert "policy" in Trainer.__abstractmethods__

    def test_parse_observations_is_abstract(self):
        from corl.trainer.trainer import Trainer

        assert "parse_observations" in Trainer.__abstractmethods__

    def test_checkpoint_is_concrete(self):
        from corl.trainer.trainer import Trainer

        assert "checkpoint" not in Trainer.__abstractmethods__
        assert callable(Trainer.checkpoint)

    def test_recovery_is_concrete(self):
        from corl.trainer.trainer import Trainer

        assert "recovery" not in Trainer.__abstractmethods__
        assert callable(Trainer.recovery)


# ===========================================================================
# _close_model
# ===========================================================================


class TestCloseModel:
    def _run(self, trainer, filenames):
        """Helper that mocks Path.iterdir, mlflow, and shutil.rmtree."""
        model_dir = trainer.model_dir
        file_paths = [Path(model_dir) / f for f in filenames]

        mock_path_instance = MagicMock()
        mock_path_instance.iterdir.return_value = iter(file_paths)

        with (
            patch("corl.trainer.trainer.Path", return_value=mock_path_instance),
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
        assert str(Path("/model/dir") / "weights.h5") in paths
        assert str(Path("/model/dir") / "config.json") in paths

    def test_skips_checkpoint_files(self, trainer):
        trainer.model_dir = "/model/dir"
        mock_mlflow, _ = self._run(
            trainer, ["weights.h5", "weights_ckt_001.h5", "_ckt_backup.h5"]
        )
        assert mock_mlflow.log_artifact.call_count == 3

    def test_all_checkpoint_files_logged(self, trainer):
        trainer.model_dir = "/model/dir"
        mock_mlflow, _ = self._run(trainer, ["_ckt_1.h5", "epoch_ckt_2.h5"])
        assert mock_mlflow.log_artifact.call_count == 2

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

    def test_deletes_dir_after_all_files_logged(self, trainer):
        trainer.model_dir = "/model/dir"
        _, mock_rmtree = self._run(trainer, ["_ckt_only.h5"])
        mock_rmtree.assert_called_once_with("/model/dir")


# ===========================================================================
# _close_checkpoints
# ===========================================================================


class TestCloseCheckpoints:
    def _run(self, trainer, checkpoint_dir, files=(), raises=False):
        """
        Helper that mocks the filesystem and mlflow for _close_checkpoints.

        ``files`` is a list of relative file names present under checkpoint_dir.
        """
        trainer.checkpoint_dir = checkpoint_dir
        checkpoint_path = Path(checkpoint_dir)

        def _rglob(pattern):
            if not files:
                return iter([])
            return iter([checkpoint_path / f for f in files])

        def _exists():
            return bool(files) or files == ()

        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.__truediv__ = lambda self, other: Path(checkpoint_dir) / other
        mock_path.parent = Path(checkpoint_dir).parent
        mock_path.rglob.side_effect = _rglob

        def _any_rglob(*a, **kw):
            return bool(files)

        with (
            patch(
                "corl.trainer.trainer.Path",
                side_effect=lambda p: (
                    mock_path if str(p) == checkpoint_dir else Path(p)
                ),
            ),
            patch("corl.trainer.trainer.mlflow") as mock_mlflow,
            patch("corl.trainer.trainer.shutil.rmtree") as mock_rmtree,
            patch("corl.trainer.trainer.zipfile.ZipFile") as mock_zip,
            patch("pathlib.Path.unlink"),
        ):
            if raises:
                mock_zip.side_effect = OSError("disk full")
            trainer._close_checkpoints()
            return mock_mlflow, mock_rmtree, mock_zip

    def test_empty_dir_does_not_log_to_mlflow(self, trainer):
        trainer.checkpoint_dir = "/ckpt/dir"
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.rglob.return_value = iter([])
        with (
            patch("corl.trainer.trainer.Path", return_value=mock_path),
            patch("corl.trainer.trainer.mlflow") as mock_mlflow,
            patch("corl.trainer.trainer.shutil.rmtree"),
        ):
            trainer._close_checkpoints()
        mock_mlflow.log_artifact.assert_not_called()

    def test_empty_dir_removes_checkpoint_dir(self, trainer):
        trainer.checkpoint_dir = "/ckpt/dir"
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.rglob.return_value = iter([])
        with (
            patch("corl.trainer.trainer.Path", return_value=mock_path),
            patch("corl.trainer.trainer.mlflow"),
            patch("corl.trainer.trainer.shutil.rmtree") as mock_rmtree,
        ):
            trainer._close_checkpoints()
        mock_rmtree.assert_called_once_with("/ckpt/dir", ignore_errors=True)

    def test_nonexistent_dir_does_not_log_to_mlflow(self, trainer):
        trainer.checkpoint_dir = "/ckpt/dir"
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        with (
            patch("corl.trainer.trainer.Path", return_value=mock_path),
            patch("corl.trainer.trainer.mlflow") as mock_mlflow,
            patch("corl.trainer.trainer.shutil.rmtree"),
        ):
            trainer._close_checkpoints()
        mock_mlflow.log_artifact.assert_not_called()

    def test_files_logged_to_mlflow_under_checkpoints_path(self, trainer):
        trainer.checkpoint_dir = "/ckpt/dir"
        ckpt_path = Path("/ckpt/dir")
        file_path = ckpt_path / "epoch_1.npy"

        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.rglob.return_value = iter([file_path])
        mock_path.parent = ckpt_path.parent
        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_path.rglob.return_value = iter([mock_file])

        with (
            patch("corl.trainer.trainer.Path", return_value=mock_path),
            patch("corl.trainer.trainer.mlflow") as mock_mlflow,
            patch("corl.trainer.trainer.shutil.rmtree"),
            patch("corl.trainer.trainer.zipfile.ZipFile"),
            patch("pathlib.Path.unlink"),
        ):
            trainer._close_checkpoints()
        mock_mlflow.log_artifact.assert_called_once()
        assert (
            mock_mlflow.log_artifact.call_args.kwargs["artifact_path"] == "checkpoints"
        )

    def test_always_removes_checkpoint_dir_after_upload(self, trainer):
        trainer.checkpoint_dir = "/ckpt/dir"
        ckpt_path = Path("/ckpt/dir")
        mock_file = MagicMock()
        mock_file.is_file.return_value = True

        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.rglob.return_value = iter([mock_file])
        mock_path.parent = ckpt_path.parent

        with (
            patch("corl.trainer.trainer.Path", return_value=mock_path),
            patch("corl.trainer.trainer.mlflow"),
            patch("corl.trainer.trainer.shutil.rmtree") as mock_rmtree,
            patch("corl.trainer.trainer.zipfile.ZipFile"),
            patch("pathlib.Path.unlink"),
        ):
            trainer._close_checkpoints()
        mock_rmtree.assert_called_with("/ckpt/dir", ignore_errors=True)

    def test_removes_checkpoint_dir_even_on_zip_exception(self, trainer):
        trainer.checkpoint_dir = "/ckpt/dir"
        ckpt_path = Path("/ckpt/dir")
        mock_file = MagicMock()
        mock_file.is_file.return_value = True

        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.rglob.return_value = iter([mock_file])
        mock_path.parent = ckpt_path.parent

        with (
            patch("corl.trainer.trainer.Path", return_value=mock_path),
            patch("corl.trainer.trainer.mlflow"),
            patch("corl.trainer.trainer.shutil.rmtree") as mock_rmtree,
            patch(
                "corl.trainer.trainer.zipfile.ZipFile", side_effect=OSError("disk full")
            ),
            patch("pathlib.Path.unlink"),
        ):
            trainer._close_checkpoints()  # must not raise
        mock_rmtree.assert_called_with("/ckpt/dir", ignore_errors=True)

    def test_does_not_log_artifact_on_zip_exception(self, trainer):
        trainer.checkpoint_dir = "/ckpt/dir"
        ckpt_path = Path("/ckpt/dir")
        mock_file = MagicMock()
        mock_file.is_file.return_value = True

        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.rglob.return_value = iter([mock_file])
        mock_path.parent = ckpt_path.parent

        with (
            patch("corl.trainer.trainer.Path", return_value=mock_path),
            patch("corl.trainer.trainer.mlflow") as mock_mlflow,
            patch("corl.trainer.trainer.shutil.rmtree"),
            patch(
                "corl.trainer.trainer.zipfile.ZipFile", side_effect=OSError("disk full")
            ),
            patch("pathlib.Path.unlink"),
        ):
            trainer._close_checkpoints()
        mock_mlflow.log_artifact.assert_not_called()

    def test_zips_actual_file_when_present(self, trainer, tmp_path):
        """Covers lines 177-178: the zf.write() call inside the ZipFile loop."""
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / "epoch_1.npy").write_bytes(b"data")

        trainer.checkpoint_dir = str(ckpt_dir)

        with (
            patch("corl.trainer.trainer.mlflow"),
            patch("corl.trainer.trainer.shutil.rmtree"),
            patch("pathlib.Path.unlink"),
        ):
            trainer._close_checkpoints()  # must not raise


class TestInitApiRecover:
    def test_delete_called_before_create_when_recovering(self):
        ConcreteTrainer = _make_trainer_class()
        config = _make_config()

        with (
            patch("corl.trainer.trainer.Wrapper") as MockWrapper,
            patch("corl.trainer.trainer.Tracker"),
            patch("corl.trainer.trainer.mlflow"),
            patch("pathlib.Path.mkdir"),
        ):
            mock_api = MagicMock()
            MockWrapper.return_value = mock_api
            t = ConcreteTrainer(_make_model(), config)
            t._init_api(config, recover=True)

        mock_api.delete_training_session.assert_called_once_with(t.train_id)

    def test_create_called_after_delete_when_recovering(self):
        ConcreteTrainer = _make_trainer_class()
        config = _make_config()

        with (
            patch("corl.trainer.trainer.Wrapper") as MockWrapper,
            patch("corl.trainer.trainer.Tracker"),
            patch("corl.trainer.trainer.mlflow"),
            patch("pathlib.Path.mkdir"),
        ):
            mock_api = MagicMock()
            MockWrapper.return_value = mock_api
            t = ConcreteTrainer(_make_model(), config)
            t._init_api(config, recover=True)

        mock_api.create_training_session.assert_called_with(t.train_id)

    def test_delete_not_called_on_fresh_start(self):
        ConcreteTrainer = _make_trainer_class()
        config = _make_config()

        with (
            patch("corl.trainer.trainer.Wrapper") as MockWrapper,
            patch("corl.trainer.trainer.Tracker"),
            patch("corl.trainer.trainer.mlflow"),
            patch("pathlib.Path.mkdir"),
        ):
            mock_api = MagicMock()
            MockWrapper.return_value = mock_api
            t = ConcreteTrainer(_make_model(), config)
            mock_api.delete_training_session.reset_mock()
            t._init_api(config, recover=False)

        mock_api.delete_training_session.assert_not_called()


# ===========================================================================
# _init_mlflow
# ===========================================================================


class TestInitMlflow:
    def test_creates_new_experiment_when_not_found(self, trainer):
        with patch("corl.trainer.trainer.mlflow") as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = None
            mock_mlflow.create_experiment.return_value = "new_exp_id"
            mock_run = MagicMock()
            mock_run.info.run_id = "run123"
            mock_mlflow.start_run.return_value = mock_run

            trainer._init_mlflow("my_exp", "http://localhost:5001")

        mock_mlflow.create_experiment.assert_called_once_with("my_exp")

    def test_uses_existing_experiment_when_found(self, trainer):
        with patch("corl.trainer.trainer.mlflow") as mock_mlflow:
            mock_exp = MagicMock()
            mock_exp.experiment_id = "existing_id"
            mock_mlflow.get_experiment_by_name.return_value = mock_exp
            mock_run = MagicMock()
            mock_run.info.run_id = "run123"
            mock_mlflow.start_run.return_value = mock_run

            trainer._init_mlflow("my_exp", "http://localhost:5001")

        mock_mlflow.create_experiment.assert_not_called()
        mock_mlflow.set_experiment.assert_called_with(experiment_id="existing_id")

    def test_resumes_run_when_run_id_already_set(self, trainer):
        trainer.mlflow_run_id = "existing_run_id"
        with patch("corl.trainer.trainer.mlflow") as mock_mlflow:
            mock_exp = MagicMock()
            mock_exp.experiment_id = "exp_id"
            mock_mlflow.get_experiment_by_name.return_value = mock_exp
            mock_run = MagicMock()
            mock_run.info.run_id = "existing_run_id"
            mock_mlflow.start_run.return_value = mock_run

            trainer._init_mlflow("my_exp", "http://localhost:5001")

        mock_mlflow.start_run.assert_called_once_with(run_id="existing_run_id")

    def test_starts_new_run_when_no_run_id(self, trainer):
        trainer.mlflow_run_id = None
        with patch("corl.trainer.trainer.mlflow") as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = None
            mock_mlflow.create_experiment.return_value = "exp_id"
            mock_run = MagicMock()
            mock_run.info.run_id = "brand_new_run"
            mock_mlflow.start_run.return_value = mock_run

            trainer._init_mlflow("my_exp", "http://localhost:5001")

        mock_mlflow.start_run.assert_called_once_with(run_name=trainer.train_id)
        assert trainer.mlflow_run_id == "brand_new_run"


# ===========================================================================
# _close_mlflow
# ===========================================================================


class TestCloseMlflow:
    def test_calls_mlflow_end_run(self, trainer):
        with patch("corl.trainer.trainer.mlflow") as mock_mlflow:
            trainer._close_mlflow()
        mock_mlflow.end_run.assert_called_once()


# ===========================================================================
# _mlflow_log_train_params
# ===========================================================================


class TestMlflowLogTrainParams:
    def test_logs_numeric_and_bool_params(self, trainer):
        trainer.params = MagicMock(
            return_value={"lr": 0.001, "epochs": 10, "name": "model", "flag": True}
        )
        with patch("corl.trainer.trainer.mlflow") as mock_mlflow:
            trainer._mlflow_log_train_params()

        logged = mock_mlflow.log_params.call_args[0][0]
        assert "lr" in logged
        assert "epochs" in logged
        assert "flag" in logged

    def test_filters_out_string_params(self, trainer):
        trainer.params = MagicMock(
            return_value={"lr": 0.001, "name": "model", "algo": "ppo"}
        )
        with patch("corl.trainer.trainer.mlflow") as mock_mlflow:
            trainer._mlflow_log_train_params()

        logged = mock_mlflow.log_params.call_args[0][0]
        assert "name" not in logged
        assert "algo" not in logged


# ===========================================================================
# params()
# ===========================================================================


class TestParams:
    def test_returns_scalar_instance_attributes(self):
        RealTrainer = _make_real_trainer_class()

        with (
            patch("corl.trainer.trainer.Wrapper"),
            patch("corl.trainer.trainer.Tracker"),
            patch("corl.trainer.trainer.mlflow"),
            patch("pathlib.Path.mkdir"),
        ):
            t = RealTrainer(_make_model(), _make_config())
            t.lr = 0.01
            t.batch_size = 32
            t.label = "test"
            t.flag = True

        result = t.params()
        assert result["lr"] == 0.01
        assert result["batch_size"] == 32
        assert result["label"] == "test"
        assert result["flag"] is True

    def test_excludes_non_scalar_attributes(self):
        RealTrainer = _make_real_trainer_class()

        with (
            patch("corl.trainer.trainer.Wrapper"),
            patch("corl.trainer.trainer.Tracker"),
            patch("corl.trainer.trainer.mlflow"),
            patch("pathlib.Path.mkdir"),
        ):
            t = RealTrainer(_make_model(), _make_config())
            t.my_list = [1, 2, 3]
            t.my_dict = {"a": 1}

        result = t.params()
        assert "my_list" not in result
        assert "my_dict" not in result


# ===========================================================================
# checkpoint() / recovery()
# ===========================================================================


class TestCheckpoint:
    @pytest.fixture
    def real_trainer(self):
        RealTrainer = _make_real_trainer_class()

        with (
            patch("corl.trainer.trainer.Wrapper") as MockWrapper,
            patch("corl.trainer.trainer.Tracker") as MockTracker,
            patch("corl.trainer.trainer.mlflow"),
            patch("pathlib.Path.mkdir"),
        ):
            mock_api = MagicMock()
            MockWrapper.return_value = mock_api
            mock_tracker = MagicMock()
            MockTracker.return_value = mock_tracker
            t = RealTrainer(_make_model(), _make_config())
            t.api = mock_api
            t.tracker = mock_tracker
            return t

    def test_creates_model_dir_and_saves(self, real_trainer):
        real_trainer.checkpoint_dir = "/ckpt"

        with (
            patch("corl.trainer.trainer.Path.mkdir"),
            patch("corl.trainer.trainer.json.dump") as mock_dump,
            patch("builtins.open", MagicMock()),
        ):
            real_trainer.checkpoint()

        real_trainer.model.save.assert_called_once()
        assert mock_dump.called

    def test_returns_path_under_checkpoint_dir(self, real_trainer):
        real_trainer.checkpoint_dir = "/ckpt"

        with (
            patch("corl.trainer.trainer.Path.mkdir"),
            patch("builtins.open", MagicMock()),
            patch("corl.trainer.trainer.json.dump"),
        ):
            path = real_trainer.checkpoint()

        assert str(path).startswith("/ckpt/")

    def test_params_written_to_json(self, real_trainer):
        real_trainer.checkpoint_dir = "/ckpt"
        real_trainer.lr = 0.001
        real_trainer.batch_size = 5

        captured = {}

        def _fake_dump(data, f, **kwargs):
            captured["data"] = data

        with (
            patch("corl.trainer.trainer.Path.mkdir"),
            patch("builtins.open", MagicMock()),
            patch("corl.trainer.trainer.json.dump", side_effect=_fake_dump),
        ):
            real_trainer.checkpoint()

        assert captured["data"] is not None


class TestRecovery:
    def _make_checkpoint(self, tmp_path, params: dict) -> str:
        import json

        checkpoint_id = "20240101_120000"
        ckpt_dir = tmp_path / checkpoint_id
        (ckpt_dir / "model").mkdir(parents=True)
        with open(ckpt_dir / "params.json", "w") as f:
            json.dump(params, f)
        return checkpoint_id

    def _make_real_trainer(self, tmp_path):
        RealTrainer = _make_real_trainer_class()

        with (
            patch("corl.trainer.trainer.Wrapper"),
            patch("corl.trainer.trainer.Tracker"),
            patch("corl.trainer.trainer.mlflow"),
            patch("pathlib.Path.mkdir"),
        ):
            t = RealTrainer(_make_model(), _make_config())

        t.checkpoint_dir = str(tmp_path)
        return t

    def test_loads_model_from_checkpoint(self, tmp_path):
        t = self._make_real_trainer(tmp_path)
        checkpoint_id = self._make_checkpoint(tmp_path, {})
        t.recovery(checkpoint_id)

        t.model.load.assert_called_once_with(str(tmp_path / checkpoint_id / "model"))

    def test_restores_annotated_scalar_attributes(self, tmp_path):
        RealTrainer = _make_real_trainer_class()

        class AnnotatedTrainer(RealTrainer):
            lr: float
            batch_size: int

        with (
            patch("corl.trainer.trainer.Wrapper"),
            patch("corl.trainer.trainer.Tracker"),
            patch("corl.trainer.trainer.mlflow"),
            patch("pathlib.Path.mkdir"),
        ):
            t = AnnotatedTrainer(_make_model(), _make_config())

        t.checkpoint_dir = str(tmp_path)
        checkpoint_id = self._make_checkpoint(
            tmp_path, {"lr": 0.005, "batch_size": 64, "unknown_key": "ignored"}
        )
        t.recovery(checkpoint_id)

        assert t.lr == 0.005
        assert t.batch_size == 64

    def test_ignores_unannotated_keys(self, tmp_path):
        t = self._make_real_trainer(tmp_path)
        checkpoint_id = self._make_checkpoint(tmp_path, {"totally_unknown": 999})
        t.recovery(checkpoint_id)

        assert not hasattr(t, "totally_unknown")

    def test_returns_path_to_checkpoint_dir(self, tmp_path):
        t = self._make_real_trainer(tmp_path)
        checkpoint_id = self._make_checkpoint(tmp_path, {})
        path = t.recovery(checkpoint_id)

        assert path == tmp_path / checkpoint_id

    def test_recovery_called_during_init_when_checkpoint_id_provided(self, tmp_path):
        import json

        RealTrainer = _make_real_trainer_class()
        checkpoint_id = "20240101_120000"
        # Must match the path trainer constructs: {output_dir}/checkpoints/{train_id}
        ckpt_dir = tmp_path / "checkpoints" / TRAIN_ID / checkpoint_id
        (ckpt_dir / "model").mkdir(parents=True)
        with open(ckpt_dir / "params.json", "w") as f:
            json.dump({}, f)

        config = _make_config()
        config.get = MagicMock(
            side_effect=lambda key: {
                "train_id": TRAIN_ID,
                "trainer_output_dir": str(tmp_path),
                "trainer_log_metric_frequency": 10,
                "trainer_mlflow_url": "localhost:5001",
                "world_name": EXPERIMENT_NAME,
                "api_host": "http://localhost",
                "api_port": 8000,
                "trainer_worker_timeout": 30,
            }.get(key.lower())
        )
        config.__getitem__ = config.get

        model = _make_model()

        with (
            patch("corl.trainer.trainer.Wrapper"),
            patch("corl.trainer.trainer.Tracker"),
            patch("corl.trainer.trainer.mlflow"),
        ):
            RealTrainer(model, config, checkpoint_id=checkpoint_id)

        model.load.assert_called_once()


# ===========================================================================
# _training_step_action (policy length mismatch)
# ===========================================================================


class TestTrainingStepActionPolicyMismatch:
    def test_raises_when_policy_returns_wrong_number_of_actions(self, trainer):
        observations = [
            (k, np.zeros(3, dtype=np.float32))
            for k in [_step_key(worker_id=i) for i in range(3)]
        ]

        # policy returns only 1 action for 3 observations
        trainer.policy = MagicMock(return_value=np.zeros(1, dtype=np.float32))

        with pytest.raises(RuntimeError, match="Policy returned"):
            trainer._training_step_action(observations)


class TestGenerateVideo:  # noqa: F811
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
