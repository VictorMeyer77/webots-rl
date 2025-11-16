import argparse
import os
from random import randint

WEBOTS_BIN_PATH = "/Applications/Webots.app/Contents/MacOS/webots"
WORLDS_PATH = "webots-rl/worlds"


def create(world: str, controller: str, train: bool) -> str:
    """
    Creates a temporary world file by replacing placeholders in the base world file
    with the specified controller and supervisor. Sets the TRAIN environment variable.

    Args:
        world (str): Base world file name (without extension).
        controller (str): Controller name or path.
        train (bool): Whether to enable training mode.

    Returns:
        str: Path to the generated temporary world file.
    """
    os.environ["TRAIN"] = "1" if train else "0"

    with open(f"{WORLDS_PATH}/{world}.wbt", "r") as world_file:
        world_content = world_file.read()
        world_content = world_content.replace("{{CONTROLLER}}", controller)
        world_content = world_content.replace("{{SUPERVISOR}}", f"{controller}_supervisor")
        with open(f"{WORLDS_PATH}/{world}_run_{randint(100000, 999999)}.wbt", "w") as tmp_file:
            tmp_file.write(world_content)
            tmp_file_path = tmp_file.name

    return tmp_file_path


def run(world_path: str, fast: bool, render: bool, port: int = None) -> None:
    """
    Runs the Webots simulation with the specified world file and options.
    Removes the temporary world file after execution.

    Args:
        world_path (str): Path to the world file to run.
        fast (bool): Whether to run in fast mode.
        render (bool): Whether to enable rendering.
    """
    webots_option = " --batch "
    webots_option += " --mode=fast " if fast else ""
    webots_option += " --no-rendering " if not render else ""
    webots_option += f" --port={port} " if port is not None else ""
    os.system(f"{WEBOTS_BIN_PATH} {world_path} {webots_option}")  # todo specify port
    os.remove(world_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation with specified world, controller, and supervisor.")
    parser.add_argument("--world", type=str, required=True, help="Path to the world file")
    parser.add_argument("--controller", type=str, required=True, help="Controller name or path")
    parser.add_argument("--port", type=int, required=False, help="Controller name or path")
    parser.add_argument("--train", action="store_true", help="Enable training mode")
    parser.add_argument("--fast", action="store_true", help="Fast running mode")
    parser.add_argument("--render", action="store_true", help="Rendering")
    args = parser.parse_args()
    world_path = create(args.world, args.controller, args.train)
    run(world_path, fast=args.fast, render=args.render, port=args.port)
