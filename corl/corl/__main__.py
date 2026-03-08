"""
CLI entry point for corl.

This module provides the command-line interface for the corl Webots reinforcement
learning toolkit. It handles argument parsing, world file generation, subprocess
management for Webots instances, and cleanup of generated artifacts.

Usage:
    corl --world <world> --controller <controller> [--worker <n>] [--trainer] [--fast]
    python -m corl --world <world> --controller <controller> [--worker <n>] [--trainer] [--fast]

Modes:
    Default (no mode flag):
        Launches a single Webots instance for the given world and controller and
        waits for it to exit. Useful for manual runs and debugging.

    --worker <n>:
        Registers a training session and spawns *n* parallel Webots worker processes
        against the Backtrain API. Workers with ID != 0 run headless (no rendering).
        All workers are awaited before the process exits.

    --trainer:
        Resolves and executes the Python trainer script located at
        ``<experiments_dir>/<world>/trainer/<controller>.py``. The trainer is
        responsible for driving the learning loop via the Backtrain API.
"""

import argparse
import logging
import socket
import subprocess
import sys
import uuid
from pathlib import Path

from corl.api.wrapper import Wrapper
from corl.utils.config import Config
from corl.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Construct and return the argument parser for the corl CLI.

    Defines the following arguments:

    Positional / required:
        --world (str):       Name of the Webots world to run, without the ``.wbt``
                             extension. Must correspond to a folder inside
                             ``experiments_dir``.
        --controller (str):  Name or path of the controller to inject into the
                             generated world file.

    Optional:
        --fast (flag):       Run Webots in fast mode (no real-time synchronisation).

    Mutually exclusive mode flags (at most one may be supplied):
        --worker (int):      Number of parallel Webots worker processes to spawn for
                             a training session.
        --trainer (flag):    Execute the Python trainer script for the given
                             world/controller pair.

    Returns:
        argparse.ArgumentParser: Configured parser ready to call ``parse_args()`` on.
    """
    parser = argparse.ArgumentParser(
        prog="corl",
        description="corl – Webots reinforcement learning toolkit",
    )

    parser.add_argument(
        "--world", type=str, default=None, help="World name (without .wbt extension)"
    )
    parser.add_argument(
        "--controller", type=str, default=None, help="Controller name or path"
    )
    parser.add_argument("--fast", action="store_true", help="Run Webots in fast mode")

    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--worker",
        type=int,
        default=None,
        help="Launch training simulation mode with the specified number of workers",
    )
    mode_group.add_argument(
        "--trainer", action="store_true", help="Launch trainer module mode"
    )

    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """
    Validate parsed CLI arguments and abort with a usage message on failure.

    Validation rules:
        - ``--world`` and ``--controller`` are both required; if either is missing
          the parser prints the usage string and exits with code 2.
        - ``--worker``, when supplied, must be a positive integer (≥ 1); passing
          ``0`` or a negative value is rejected.

    Args:
        args (argparse.Namespace): The namespace returned by ``parser.parse_args()``.
        parser (argparse.ArgumentParser): The parser used to parse ``args``, used to
            call ``parser.error()`` so that failures produce a consistent usage
            message and a non-zero exit code.

    Returns:
        None

    Raises:
        SystemExit: Via ``parser.error()`` if any validation rule is violated.
    """
    if not args.world or not args.controller:
        parser.error("--world and --controller are required")

    if args.worker is not None and args.worker < 1:
        parser.error("--worker must be a positive integer")


def _get_free_port() -> int:
    """
    Return a free TCP port on the loopback interface chosen by the OS.

    Binds a temporary socket to ``('127.0.0.1', 0)``, reads back the port that
    the OS assigned, then immediately releases the socket. The returned port is
    free at the moment of the call but is not reserved — callers should use it
    promptly to minimise the chance of a race condition.

    Returns:
        int: An available port number in the range 1024–65535.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _create_world(experiments_dir: str, world: str, controller: str) -> Path:
    """
    Instantiate a world file from a template and return its path.

    Reads the template at ``<experiments_dir>/<world>/worlds/<world>.wbt``,
    replaces every occurrence of the placeholder ``{{CONTROLLER}}`` with
    *controller*, and writes the result to a new file with a unique 8-character
    hex suffix (e.g. ``<world>_run_3f2a1b4c.wbt``) in the same directory.

    The unique suffix prevents concurrent runs from overwriting each other's
    world files and makes it easy to identify orphaned files after a crash.

    Args:
        experiments_dir (str): Root directory that contains per-world project folders.
        world (str): Name of the world (without ``.wbt``); must match a subdirectory
            inside *experiments_dir*.
        controller (str): Controller name or path to inject in place of
            ``{{CONTROLLER}}`` inside the template.

    Returns:
        Path: Absolute path to the newly created world file.
    """
    world_dir = Path(experiments_dir) / world / "worlds"
    template = world_dir / f"{world}.wbt"
    world_file = world_dir / f"{world}_run_{uuid.uuid4().hex[:8]}.wbt"
    content = template.read_text()
    world_file.write_text(content.replace("{{CONTROLLER}}", controller))
    logger.debug(
        f"Created world file '{world_file}' from template '{template}' with controller '{controller}'"
    )
    return world_file


def _launch_webots(
    bin_path: str, world_path: Path, fast: bool, env: dict[str, str]
) -> subprocess.Popen:
    """
    Start a Webots subprocess and return its process handle.

    Builds the Webots command from *bin_path*, *world_path*, and a randomly
    selected free port, then spawns the process with the given environment.

    Rendering behaviour is determined by the ``WEBOTS_WORKER_ID`` key in *env*:
        - Worker ID ``"0"`` (or absent): rendered instance, used as the witness
          that captures video.
        - Any other worker ID: ``--no-rendering`` is appended so the instance
          runs fully headless, reducing resource usage on parallel workers.

    Args:
        bin_path (str): Absolute path to the Webots executable.
        world_path (Path): Path to the ``.wbt`` world file to load.
        fast (bool): When ``True``, appends ``--mode=fast`` to disable real-time
            synchronisation and run the simulation as fast as possible.
        env (dict[str, str]): Full environment dictionary passed to the child
            process. Must contain ``WEBOTS_WORKER_ID`` if rendering should be
            suppressed for non-zero workers.

    Returns:
        subprocess.Popen: Handle to the running Webots process. The caller is
            responsible for calling ``.wait()`` or ``.terminate()``.
    """
    command = [bin_path, str(world_path), "--batch", f"--port={_get_free_port()}"]

    if fast:
        command.append("--mode=fast")
    if env.get("WEBOTS_WORKER_ID", "0") != "0":
        command.append("--no-rendering")

    logger.debug(f"Launching Webots: {' '.join(command)}")
    return subprocess.Popen(command, env=env)


def _launch_training_workers(
    config: Config,
    world_path: Path,
    worker: int,
) -> list[subprocess.Popen]:
    """
    Spawn *worker* parallel Webots worker processes and return their handles.

    Registers each worker with the Backtrain API via :class:`~corl.api.wrapper.Wrapper`,
    builds a per-worker environment that includes the assigned ``WEBOTS_WORKER_ID``,
    and launches a Webots subprocess for each worker in fast/headless mode.

    Args:
        config (Config): Loaded application configuration. Must have ``TRAIN_ID``,
            ``bin_path``, and all other keys required by
            :func:`_launch_webots` already set.
        world_path (Path): Path to the generated ``.wbt`` world file that every
            worker will load.
        worker (int): Number of worker processes to spawn. Must be ≥ 1.

    Returns:
        list[subprocess.Popen]: Handles to all spawned Webots worker processes,
            in the order they were started. The caller is responsible for
            calling ``.wait()`` on each handle.
    """
    api = Wrapper(config)
    processes: list[subprocess.Popen] = []
    for _ in range(worker):
        worker_id = api.add_worker(config.get("train_id"))
        worker_env = {**config.environ(), "WEBOTS_WORKER_ID": str(worker_id)}
        logger.info(
            f"Starting worker {worker_id} for training session '{config.get('train_id')}'"
        )
        processes.append(
            _launch_webots(config.get("bin_path"), world_path, True, worker_env)
        )

    return processes


def _launch_trainer(
    experiments_dir: str, world: str, controller: str, env: dict[str, str]
) -> subprocess.Popen:
    """
    Launch the trainer script for the given world/controller pair.

    Resolves the trainer script at
    ``<experiments_dir>/<world>/trainer/<controller>.py`` and executes it with
    the current Python interpreter, passing *env* as the child-process
    environment.

    Args:
        experiments_dir (str): Root directory that contains per-world project folders.
        world (str): Name of the world; used to locate the trainer subdirectory.
        controller (str): Name of the controller; used as the trainer script filename
            (without ``.py``).
        env (dict[str, str]): Full environment dictionary passed to the child
            process. Typically built from ``config.environ()`` with training
            identifiers already injected.

    Returns:
        subprocess.Popen: Handle to the running trainer process. The caller is
            responsible for calling ``.wait()`` or ``.terminate()``.

    Raises:
        FileNotFoundError: If the trainer script does not exist at the expected path.
    """
    trainer_path = Path(experiments_dir) / world / f"trainer/{controller}.py"
    if not trainer_path.exists():
        raise FileNotFoundError(f"Trainer script '{trainer_path}' does not exist")
    process = subprocess.Popen([sys.executable, str(trainer_path)], env=env)
    logger.info(f"Launched trainer module '{trainer_path}' with PID {process.pid}")
    return process


def _run_single(config: Config, world_path: Path, fast: bool) -> None:
    """
    Launch a single Webots instance for a non-training run and block until it exits.

    Sets ``LOG_FILE_NAME`` to ``"run"`` on *config* so that any file logging
    produced during this session is written to a clearly identifiable file,
    then delegates to :func:`_launch_webots` and waits for the process to finish.

    Args:
        config (Config): Loaded application configuration. Must have ``bin_path``
            and all environment keys required by :func:`_launch_webots`.
        world_path (Path): Path to the generated ``.wbt`` world file to load.
        fast (bool): When ``True``, Webots is started with ``--mode=fast`` to
            disable real-time synchronisation.
    """
    config.set("LOG_FILE_NAME", "run")
    _launch_webots(config.get("bin_path"), world_path, fast, config.environ()).wait()


def _remove_world(world_path: Path) -> None:
    """
    Delete the generated world file and its associated ``.wbproj`` file.

    Webots automatically creates a hidden project file alongside every ``.wbt``
    file (e.g. ``.my_world_run_3f2a1b4c.wbproj``). This function removes both
    the world file and that sidecar file so that the experiments directory does
    not accumulate stale artifacts between runs.

    Failures are non-fatal: if either file cannot be removed (e.g. because it
    was already deleted by another process) a warning is logged and execution
    continues normally.

    Args:
        world_path (Path): Path to the generated ``.wbt`` world file to remove.
            The matching ``.wbproj`` path is derived automatically from this value.
    """
    world_file_name = world_path.name
    proj_file_name = "." + world_file_name.replace(".wbt", ".wbproj")
    proj_path = world_path.parent / proj_file_name

    try:
        world_path.unlink(missing_ok=True)
        proj_path.unlink(missing_ok=True)
        logger.debug(f"Removed world file '{world_path}'")
    except OSError as e:
        logger.warning(f"Failed to remove world file '{world_path}': {e}")


def main() -> None:
    """
    CLI entry point — orchestrates argument parsing, dispatching, and cleanup.

    Execution order:
        1. Build and parse CLI arguments via :func:`_build_arg_parser`.
        2. Validate the parsed arguments via :func:`_validate_args`.
        3. Load application configuration and set up logging.
        4. Generate a uniquely named world file via :func:`_create_world`.
        5. Dispatch to the appropriate mode:
               - ``--worker <n>``: set ``TRAIN_ID`` on config, spawn *n* training
                 workers via :func:`_launch_training_workers`, and wait for all
                 of them to finish.
               - ``--trainer``: set ``TRAIN_ID`` on config, start the trainer
                 subprocess via :func:`_launch_trainer`, and wait for it.
               - *(default)*: run a single Webots instance via :func:`_run_single`
                 and wait for it.
        6. Remove the generated world file via :func:`_remove_world`.

    Raises:
        SystemExit: If argument validation fails (via ``parser.error()``).
        FileNotFoundError: If the world template or trainer script cannot be found.
        Exception: Any unhandled exception from the dispatched mode propagates to
            the caller / interpreter.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    config = Config()
    config.set("world_name", args.world)
    experiments_dir = config.get("experiments_dir")
    setup_logging(config)

    train_id = f"{args.world}_{args.controller}"
    world_path = _create_world(experiments_dir, args.world, args.controller)

    if args.worker is not None:
        config.set("TRAIN_ID", train_id)
        processes = _launch_training_workers(config, world_path, args.worker)
        for process in processes:
            process.wait()
    elif args.trainer:
        config.set("TRAIN_ID", train_id)
        _launch_trainer(
            experiments_dir, args.world, args.controller, config.environ()
        ).wait()
    else:
        _run_single(config, world_path, args.fast)

    _remove_world(world_path)


if __name__ == "__main__":
    main()
