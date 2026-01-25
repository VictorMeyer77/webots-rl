# uncommented and unstable

import sys

sys.path.append("../../libraries")
sys.path.append("/Applications/Webots.app/Contents/MacOS/webots")

import logging
import os

from brain.environment.simple_arena import EnvironmentSimpleArena
from brain.multi_trainer.simple_arena.sac_discrete import TrainerAgentSACDiscreteSimpleArena
from brain.utils.logger import logger
from controller import Supervisor

# Simulation Parameters
TIME_STEP = 64  # Simulation timestep in milliseconds (15.625 Hz)
EPISODE_SIZE = 1000  # Maximum steps per episode before timeout
EPOCHS = 2000  # Number of training episodes

if __name__ == "__main__":
    # Setup logging
    logger.add_console_logger(logging.INFO)
    logger.add_file_logger(logging.INFO)

    # Determine mode from environment variable
    train = True if os.getenv("TRAIN") == "1" else False

    # Initialize Webots supervisor and environment
    supervisor = Supervisor()
    environment = EnvironmentSimpleArena(supervisor, TIME_STEP, EPISODE_SIZE)

    if train:
        tcp_port = int(os.getenv("TCP_PORT"))
        # Training mode: Create trainer and run training loop
        trainer = TrainerAgentSACDiscreteSimpleArena(environment=environment, tcp_port=tcp_port)
        trainer.run(EPOCHS)
    else:
        # Evaluation mode: Run single episode
        final_state = environment.run()
        logger().info(f"Final state: {final_state.to_json()}")

    # Cleanup
    environment.quit()
