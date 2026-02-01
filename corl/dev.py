import argparse

WEBOTS_REPO_URL = "https://github.com/cyberbotics/webots.git"
WEBOTS_REPO_TAG = "R2025a"
WEBOTS_LIB_PATH = "lib/controller/python/controller"


# Assume git is installed and available in the system PATH
# Run after uv sync
def install_webots_packages():
    import os
    import shutil
    import site
    import subprocess
    import tempfile

    venv_site_packages = site.getsitepackages()[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(
            ["git", "clone", "--no-checkout", WEBOTS_REPO_URL, temp_dir], check=True
        )

        subprocess.run(
            ["git", "sparse-checkout", "init", "--cone"], cwd=temp_dir, check=True
        )

        subprocess.run(
            ["git", "sparse-checkout", "set", WEBOTS_LIB_PATH], cwd=temp_dir, check=True
        )

        subprocess.run(
            ["git", "checkout", f"tags/{WEBOTS_REPO_TAG}"], cwd=temp_dir, check=True
        )

        python_lib_path = os.path.join(venv_site_packages, "controller")

        if os.path.exists(python_lib_path):
            shutil.rmtree(python_lib_path)

        shutil.move(
            os.path.join(temp_dir, WEBOTS_LIB_PATH),
            os.path.join(venv_site_packages, "controller"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Additional commands for developing")
    parser.add_argument("command", type=str, help="Function to execute")

    command = parser.parse_args().command
    if command == "install_webots_packages":
        install_webots_packages()
    else:
        raise ValueError(f"Unknown command '{command}'")
