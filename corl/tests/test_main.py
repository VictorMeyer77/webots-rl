"""
Unit tests for corl.__main__

All subprocess, filesystem, and config I/O are mocked — no real
Webots processes are started or files created.
"""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import corl.__main__ as main_module
from corl.__main__ import (
    _build_arg_parser,
    _create_world,
    _get_free_port,
    _launch_trainer,
    _launch_webots,
    _remove_world,
    _run_single,
    _validate_args,
)

# ---------------------------------------------------------------------------
# _build_arg_parser
# ---------------------------------------------------------------------------


class TestBuildArgParser:
    def test_returns_argument_parser(self):
        parser = _build_arg_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parses_world_and_controller(self):
        parser = _build_arg_parser()
        args = parser.parse_args(["--world", "maze", "--controller", "dqn"])
        assert args.world == "maze"
        assert args.controller == "dqn"

    def test_fast_defaults_to_false(self):
        parser = _build_arg_parser()
        args = parser.parse_args(["--world", "w", "--controller", "c"])
        assert args.fast is False

    def test_fast_flag_sets_true(self):
        parser = _build_arg_parser()
        args = parser.parse_args(["--world", "w", "--controller", "c", "--fast"])
        assert args.fast is True

    def test_worker_parsed_as_int(self):
        parser = _build_arg_parser()
        args = parser.parse_args(["--world", "w", "--controller", "c", "--worker", "4"])
        assert args.worker == 4

    def test_trainer_flag(self):
        parser = _build_arg_parser()
        args = parser.parse_args(["--world", "w", "--controller", "c", "--trainer"])
        assert args.trainer is True

    def test_worker_and_trainer_are_mutually_exclusive(self):
        parser = _build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["--world", "w", "--controller", "c", "--worker", "2", "--trainer"]
            )

    def test_env_parsed_as_path(self):
        parser = _build_arg_parser()
        args = parser.parse_args(
            ["--world", "w", "--controller", "c", "--env", "/some/.env"]
        )
        assert args.env == Path("/some/.env")

    def test_defaults_when_no_args(self):
        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.world is None
        assert args.controller is None
        assert args.worker is None
        assert args.trainer is False
        assert args.fast is False
        assert args.env is None


# ---------------------------------------------------------------------------
# _validate_args
# ---------------------------------------------------------------------------


class TestValidateArgs:
    def _make_args(self, world="maze", controller="dqn", worker=None):
        ns = argparse.Namespace(world=world, controller=controller, worker=worker)
        return ns

    def test_valid_args_does_not_raise(self):
        parser = _build_arg_parser()
        args = self._make_args()
        _validate_args(args, parser)  # should not raise

    def test_missing_world_exits(self):
        parser = _build_arg_parser()
        args = self._make_args(world=None)
        with pytest.raises(SystemExit):
            _validate_args(args, parser)

    def test_missing_controller_exits(self):
        parser = _build_arg_parser()
        args = self._make_args(controller=None)
        with pytest.raises(SystemExit):
            _validate_args(args, parser)

    def test_empty_world_exits(self):
        parser = _build_arg_parser()
        args = self._make_args(world="")
        with pytest.raises(SystemExit):
            _validate_args(args, parser)

    def test_worker_zero_exits(self):
        parser = _build_arg_parser()
        args = self._make_args(worker=0)
        with pytest.raises(SystemExit):
            _validate_args(args, parser)

    def test_worker_negative_exits(self):
        parser = _build_arg_parser()
        args = self._make_args(worker=-1)
        with pytest.raises(SystemExit):
            _validate_args(args, parser)

    def test_worker_one_is_valid(self):
        parser = _build_arg_parser()
        args = self._make_args(worker=1)
        _validate_args(args, parser)  # should not raise


# ---------------------------------------------------------------------------
# _get_free_port
# ---------------------------------------------------------------------------


class TestGetFreePort:
    def test_returns_int(self):
        port = _get_free_port()
        assert isinstance(port, int)

    def test_port_in_valid_range(self):
        port = _get_free_port()
        assert 1024 <= port <= 65535

    def test_uses_loopback_interface(self):
        with patch("corl.__main__.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_sock.__enter__ = MagicMock(return_value=mock_sock)
            mock_sock.__exit__ = MagicMock(return_value=False)
            mock_sock.getsockname.return_value = ("127.0.0.1", 54321)
            mock_socket_cls.return_value = mock_sock
            port = _get_free_port()

        mock_sock.bind.assert_called_once_with(("127.0.0.1", 0))
        assert port == 54321


# ---------------------------------------------------------------------------
# _create_world
# ---------------------------------------------------------------------------


class TestCreateWorld:
    def test_returns_path(self, tmp_path):
        world_dir = tmp_path / "maze" / "worlds"
        world_dir.mkdir(parents=True)
        template = world_dir / "maze.wbt"
        template.write_text("robot controller={{CONTROLLER}} end")

        result = _create_world(str(tmp_path), "maze", "dqn")

        assert isinstance(result, Path)
        assert result.suffix == ".wbt"
        assert result.parent == world_dir

    def test_replaces_controller_placeholder(self, tmp_path):
        world_dir = tmp_path / "maze" / "worlds"
        world_dir.mkdir(parents=True)
        (world_dir / "maze.wbt").write_text("{{CONTROLLER}}")

        result = _create_world(str(tmp_path), "maze", "dqn")

        assert result.read_text() == "dqn"

    def test_generated_file_has_unique_suffix(self, tmp_path):
        world_dir = tmp_path / "maze" / "worlds"
        world_dir.mkdir(parents=True)
        (world_dir / "maze.wbt").write_text("{{CONTROLLER}}")

        p1 = _create_world(str(tmp_path), "maze", "dqn")
        p2 = _create_world(str(tmp_path), "maze", "dqn")

        assert p1 != p2

    def test_filename_contains_world_name(self, tmp_path):
        world_dir = tmp_path / "maze" / "worlds"
        world_dir.mkdir(parents=True)
        (world_dir / "maze.wbt").write_text("{{CONTROLLER}}")

        result = _create_world(str(tmp_path), "maze", "dqn")

        assert "maze" in result.name

    def test_multiple_placeholder_replacements(self, tmp_path):
        world_dir = tmp_path / "w" / "worlds"
        world_dir.mkdir(parents=True)
        (world_dir / "w.wbt").write_text("{{CONTROLLER}} and {{CONTROLLER}}")

        result = _create_world(str(tmp_path), "w", "ctrl")

        assert result.read_text() == "ctrl and ctrl"

    def test_raises_when_template_missing(self, tmp_path):
        world_dir = tmp_path / "maze" / "worlds"
        world_dir.mkdir(parents=True)
        # No template file created

        with pytest.raises(FileNotFoundError):
            _create_world(str(tmp_path), "maze", "dqn")


# ---------------------------------------------------------------------------
# _launch_webots
# ---------------------------------------------------------------------------


class TestLaunchWebots:
    def _popen_mock(self):
        return patch("corl.__main__.subprocess.Popen")

    def test_returns_popen_handle(self):
        mock_proc = MagicMock()
        with patch("corl.__main__.subprocess.Popen", return_value=mock_proc):
            with patch("corl.__main__._get_free_port", return_value=9999):
                result = _launch_webots("/usr/bin/webots", Path("world.wbt"), False, {})
        assert result is mock_proc

    def test_includes_batch_flag(self):
        with patch("corl.__main__.subprocess.Popen") as mock_popen:
            with patch("corl.__main__._get_free_port", return_value=9999):
                _launch_webots("/usr/bin/webots", Path("world.wbt"), False, {})
        cmd = mock_popen.call_args[0][0]
        assert "--batch" in cmd

    def test_includes_port_flag(self):
        with patch("corl.__main__.subprocess.Popen") as mock_popen:
            with patch("corl.__main__._get_free_port", return_value=12345):
                _launch_webots("/usr/bin/webots", Path("world.wbt"), False, {})
        cmd = mock_popen.call_args[0][0]
        assert "--port=12345" in cmd

    def test_fast_mode_appends_flag(self):
        with patch("corl.__main__.subprocess.Popen") as mock_popen:
            with patch("corl.__main__._get_free_port", return_value=9999):
                _launch_webots("/usr/bin/webots", Path("world.wbt"), True, {})
        cmd = mock_popen.call_args[0][0]
        assert "--mode=fast" in cmd

    def test_no_fast_mode_omits_flag(self):
        with patch("corl.__main__.subprocess.Popen") as mock_popen:
            with patch("corl.__main__._get_free_port", return_value=9999):
                _launch_webots("/usr/bin/webots", Path("world.wbt"), False, {})
        cmd = mock_popen.call_args[0][0]
        assert "--mode=fast" not in cmd

    def test_non_zero_worker_id_adds_no_rendering(self):
        env = {"WEBOTS_WORKER_ID": "2"}
        with patch("corl.__main__.subprocess.Popen") as mock_popen:
            with patch("corl.__main__._get_free_port", return_value=9999):
                _launch_webots("/usr/bin/webots", Path("world.wbt"), False, env)
        cmd = mock_popen.call_args[0][0]
        assert "--no-rendering" in cmd

    def test_worker_id_zero_no_rendering_absent(self):
        env = {"WEBOTS_WORKER_ID": "0"}
        with patch("corl.__main__.subprocess.Popen") as mock_popen:
            with patch("corl.__main__._get_free_port", return_value=9999):
                _launch_webots("/usr/bin/webots", Path("world.wbt"), False, env)
        cmd = mock_popen.call_args[0][0]
        assert "--no-rendering" not in cmd

    def test_absent_worker_id_no_rendering_absent(self):
        with patch("corl.__main__.subprocess.Popen") as mock_popen:
            with patch("corl.__main__._get_free_port", return_value=9999):
                _launch_webots("/usr/bin/webots", Path("world.wbt"), False, {})
        cmd = mock_popen.call_args[0][0]
        assert "--no-rendering" not in cmd

    def test_env_passed_to_popen(self):
        env = {"MY_KEY": "value"}
        with patch("corl.__main__.subprocess.Popen") as mock_popen:
            with patch("corl.__main__._get_free_port", return_value=9999):
                _launch_webots("/usr/bin/webots", Path("world.wbt"), False, env)
        assert mock_popen.call_args[1]["env"] == env


# ---------------------------------------------------------------------------
# _remove_world
# ---------------------------------------------------------------------------


class TestRemoveWorld:
    def test_removes_wbt_file(self, tmp_path):
        wbt = tmp_path / "maze_run_abc.wbt"
        wbt.touch()
        _remove_world(wbt)
        assert not wbt.exists()

    def test_removes_wbproj_sidecar(self, tmp_path):
        wbt = tmp_path / "maze_run_abc.wbt"
        proj = tmp_path / ".maze_run_abc.wbproj"
        wbt.touch()
        proj.touch()
        _remove_world(wbt)
        assert not proj.exists()

    def test_does_not_raise_when_files_already_gone(self, tmp_path):
        wbt = tmp_path / "maze_run_abc.wbt"
        _remove_world(wbt)  # file never existed — should not raise

    def test_logs_warning_on_oserror(self, tmp_path, caplog):
        import logging

        wbt = tmp_path / "maze_run_abc.wbt"
        with patch.object(Path, "unlink", side_effect=OSError("permission denied")):
            with caplog.at_level(logging.WARNING, logger="corl.__main__"):
                _remove_world(wbt)
        assert any("Failed" in r.message for r in caplog.records)

    def test_proj_filename_derived_from_wbt_name(self, tmp_path):
        wbt = tmp_path / "my_world_run_12345678.wbt"
        proj = tmp_path / ".my_world_run_12345678.wbproj"
        wbt.touch()
        proj.touch()
        _remove_world(wbt)
        assert not proj.exists()


# ---------------------------------------------------------------------------
# _launch_trainer
# ---------------------------------------------------------------------------


class TestLaunchTrainer:
    def _make_config(self, experiments_dir: str, env: dict | None = None) -> MagicMock:
        config = MagicMock()
        config.get.side_effect = lambda key: (
            experiments_dir if key == "experiments_dir" else None
        )
        config.environ.return_value = env or {}
        return config

    def test_raises_when_script_missing(self, tmp_path):
        config = self._make_config(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _launch_trainer(config, "maze", "dqn")

    def test_launches_with_python_executable(self, tmp_path):
        trainer_dir = tmp_path / "maze" / "trainer"
        trainer_dir.mkdir(parents=True)
        (trainer_dir / "dqn.py").touch()
        config = self._make_config(str(tmp_path))

        with patch("corl.__main__.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            _launch_trainer(config, "maze", "dqn")

        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == main_module.sys.executable

    def test_script_path_in_command(self, tmp_path):
        trainer_dir = tmp_path / "maze" / "trainer"
        trainer_dir.mkdir(parents=True)
        script = trainer_dir / "dqn.py"
        script.touch()
        config = self._make_config(str(tmp_path))

        with patch("corl.__main__.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            _launch_trainer(config, "maze", "dqn")

        cmd = mock_popen.call_args[0][0]
        assert str(script) in cmd

    def test_sets_log_file_name_to_trainer(self, tmp_path):
        trainer_dir = tmp_path / "maze" / "trainer"
        trainer_dir.mkdir(parents=True)
        (trainer_dir / "dqn.py").touch()
        config = self._make_config(str(tmp_path))

        with patch("corl.__main__.subprocess.Popen", return_value=MagicMock()):
            _launch_trainer(config, "maze", "dqn")

        config.set.assert_called_with("log_file_name", "trainer")

    def test_returns_popen_handle(self, tmp_path):
        trainer_dir = tmp_path / "maze" / "trainer"
        trainer_dir.mkdir(parents=True)
        (trainer_dir / "dqn.py").touch()
        config = self._make_config(str(tmp_path))

        mock_proc = MagicMock()
        with patch("corl.__main__.subprocess.Popen", return_value=mock_proc):
            result = _launch_trainer(config, "maze", "dqn")

        assert result is mock_proc


# ---------------------------------------------------------------------------
# _run_single
# ---------------------------------------------------------------------------


class TestRunSingle:
    def test_sets_log_file_name_to_run(self):
        config = MagicMock()
        config.get.return_value = "/usr/bin/webots"
        config.environ.return_value = {}
        mock_proc = MagicMock()

        with patch("corl.__main__._launch_webots", return_value=mock_proc):
            _run_single(config, Path("world.wbt"), False)

        config.set.assert_called_with("log_file_name", "run")

    def test_waits_for_process(self):
        config = MagicMock()
        config.get.return_value = "/usr/bin/webots"
        config.environ.return_value = {}
        mock_proc = MagicMock()

        with patch("corl.__main__._launch_webots", return_value=mock_proc):
            _run_single(config, Path("world.wbt"), False)

        mock_proc.wait.assert_called_once()

    def test_passes_fast_flag(self):
        config = MagicMock()
        config.get.return_value = "/usr/bin/webots"
        config.environ.return_value = {}

        with patch("corl.__main__._launch_webots", return_value=MagicMock()) as mock_lw:
            _run_single(config, Path("world.wbt"), True)

        _, _, fast, _ = mock_lw.call_args[0]
        assert fast is True
