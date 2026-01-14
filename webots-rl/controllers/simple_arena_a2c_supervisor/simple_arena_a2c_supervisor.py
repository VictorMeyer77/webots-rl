"""
Simple Arena A2C supervisor entry point for Webots simulation.

This module serves as the main entry point for the Webots supervisor in the
Simple Arena environment using the A2C (Advantage Actor-Critic) algorithm.
It initializes the Webots simulation environment, configures the supervisor
agent, and coordinates either training or evaluation mode based on environment
variables.

The supervisor runs within the Webots world process and has privileged access
to the simulation state, including robot positions, sensor readings, and
reward computation. In training mode, it communicates with the A2C training
server via TCP sockets to coordinate policy learning across multiple parallel
environments. In evaluation mode, it runs a single episode to assess the
learned policy.

The supervisor manages the episode lifecycle: resetting the environment,
advancing simulation timesteps, computing rewards, detecting termination
conditions, and collecting performance metrics.

Constants:
    TIME_STEP (int): Webots simulation timestep in milliseconds (64ms).
        Determines the frequency of physics updates and control loops.
        64ms corresponds to approximately 15.625 Hz update rate.

    EPISODE_SIZE (int): Maximum number of steps per episode before timeout (500).
        Prevents infinite episodes and ensures training progress. With 64ms
        timesteps, this corresponds to 32 seconds of simulated time per episode.

    EPOCHS (int): Number of training episodes to execute (1000). Each epoch
        represents a complete episode from initial state to termination.
        Total training time depends on episode lengths and environment complexity.

Environment Variables:
    TRAIN (str): Controls execution mode. Set to "1" for training mode, any
        other value or unset for evaluation mode.

    TCP_PORT (str): TCP port number for communication with the A2C training
        server (training mode only). Each parallel environment uses a unique
        port assigned by the training orchestrator.
"""

import sys

sys.path.append("../../libraries")
sys.path.append("/Applications/Webots.app/Contents/MacOS/webots")

import logging
import os

from brain.environment.simple_arena import EnvironmentSimpleArena
from brain.multi_trainer.simple_arena.a2c import TrainerAgentA2CSimpleArena
from brain.utils.logger import logger
from controller import Supervisor

# Simulation Parameters
TIME_STEP = 64  # Simulation timestep in milliseconds (15.625 Hz)
EPISODE_SIZE = 500  # Maximum steps per episode before timeout
EPOCHS = 1000  # Number of training episodes

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
        trainer = TrainerAgentA2CSimpleArena(environment=environment, tcp_port=tcp_port)
        trainer.run(EPOCHS)
    else:
        # Evaluation mode: Run single episode
        final_state = environment.run()
        logger().info(f"Final state: {final_state.to_json()}")

    # Cleanup
    environment.quit()
