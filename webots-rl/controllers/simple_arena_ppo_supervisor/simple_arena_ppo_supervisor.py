"""
Simple Arena PPO Supervisor Controller

This module implements a Webots supervisor controller for training and evaluating
reinforcement learning agents using the PPO (Proximal Policy Optimization) algorithm
in a simple arena environment.

Architecture:
    The controller operates as a bridge between Webots simulation and the RL training
    pipeline. It manages the simulation lifecycle, environment state, and coordinates
    with external training processes through TCP communication.


Environment Variables:
    TRAIN : str
        Set to "1" to enable training mode, any other value enables evaluation mode
    TCP_PORT : str
        TCP port number for trainer communication (training mode only)
        Must match the port used by the external trainer process

Simulation Parameters:
    TIME_STEP : int
        Simulation timestep in milliseconds (64ms = 15.625 Hz)
        Determines the frequency of sensor readings and control updates

    EPISODE_SIZE : int
        Maximum number of steps per episode before automatic reset (500 steps)
        Prevents infinite episodes and ensures consistent training experience

    EPOCHS : int
        Number of training episodes to execute (1000)
        Total training duration = EPOCHS × EPISODE_SIZE × TIME_STEP
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
