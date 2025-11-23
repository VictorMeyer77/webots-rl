"""
Monte Carlo trainer specialization for a simple arena scenario.

This module defines `TrainerMonteCarloSimpleArena`, which extends a base Monte Carlo
trainer (`TrainerMonteCarlo`) to interact with a Webots supervisor/controller loop via
JSON message passing:
- Performs an initial synchronization handshake with the robot controller.
- Waits for an observation message containing distance sensor readings.
- Selects an action using an epsilon-greedy policy and sends it.
- Receives state and reward from the environment step.
- Stops when the episode terminates (success or failure) and restarts the controller.

Expected observation format:
{
    "distance_sensors": [int, ...]  # discrete sensor values
    ...
}

State object (from environment.step()) is assumed to expose:
- finish_line_distance: float
- is_terminated: bool
- is_success: bool
- step_index: int
"""

import random

import numpy as np
from brain.environment import Environment
from brain.trainer.monte_carlo import TrainerMonteCarlo
from brain.utils.logger import logger

OBSERVATION_SIZE = 8  # Number of distance sensors
OBSERVATION_CARDINALITY = 3  # Number of discrete bins per sensor
ACTION_SIZE = 3  # Number of discrete actions


class TrainerMonteCarloSimpleArena(TrainerMonteCarlo):
    """
    Monte Carlo control trainer for a simple arena task.

    This trainer interacts with a Webots environment using JSON messages to
    synchronize, receive observations, select actions via an epsilon-greedy policy,
    and manage episode progression.

    Methods:
        policy(observation): Selects an action using epsilon-greedy strategy.
        simulation(): Runs a single episode, handling synchronization, observation,
                      action selection, and reward collection.
    """

    def __init__(
        self,
        environment: Environment,
        model_name: str,
        gamma: float,
        epsilon: float,
    ):
        """
        Initialize the Monte Carlo trainer.

        Parameters:
            environment: Simulation environment providing reset and interaction.
            model_name: Name used for saving the Q-table model.
            gamma: Discount factor applied to future rewards.
            epsilon: Initial exploration rate for the epsilon-greedy policy.
        """
        super().__init__(
            environment=environment,
            model_name=model_name,
            action_size=ACTION_SIZE,
            observation_size=OBSERVATION_SIZE,
            observation_cardinality=OBSERVATION_CARDINALITY,
            gamma=gamma,
            epsilon=epsilon,
        )

    def policy(self, observation: dict) -> int:
        """
        Select an action via epsilon-greedy strategy.

        Parameters:
            observation: Dictionary containing key `distance_sensors`.

        Returns:
            Action index (int).
        """
        observation = observation["distance_sensors"]
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)
            logger().debug(f"Taking random action {action} for state {observation}")
        else:
            q_values = self.model.q_table[self.model.observation_to_index(observation)]
            max_q = q_values.max()
            max_indexes = np.where(q_values == max_q)[0]
            action = np.random.choice(max_indexes)
            logger().debug(f"Taking best action {action} for state {observation} with Q-value {max_q}")
        return int(action)

    def simulation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run a single episode interaction loop with the environment.

        Workflow:
            1. Synchronize with controller on first step (wait for `{"ack": 1}`).
            2. Poll for an observation message (`"observation"` key).
            3. Append sensor values to trajectory.
            4. Select and send an action (`{"action": <int>}`).
            5. Advance environment, collect state and reward.
            6. Log progress (distance to finish line).
            7. If terminal, restart controller and break.

        Returns:
            observations: ndarray of sensor vectors.
            actions: ndarray of action indices.
            rewards: ndarray of scalar rewards (type as provided by environment).
        """
        observations = []
        actions = []
        rewards = []
        queue = self.environment.queue
        state = None
        sync = False
        step_observation = None
        step_send_action = False
        step_control = False

        # Main supervisor-driven loop; exits on Webots termination (-1) or episode end.
        while self.environment.supervisor.step(self.environment.timestep) != -1:

            queue.clear_buffer()

            # (1) Initial synchronization handshake on the very first step. Randomize epuck position.
            if not sync:
                if not queue.search_message("ack"):
                    queue.send({"sync": 1})
                    logger().debug("Sent sync message to controller.")
                    continue
                else:
                    sync = True
                    self.environment.randomize()
                    logger().debug("Synchronization with controller successful.")

            # (2) Blocking wait for an observation message.
            if step_observation is None:
                observation_messages = queue.search_message("observation")
                if not observation_messages:
                    continue
                else:
                    step_observation = observation_messages[0]["observation"]
                    observations.append(step_observation["distance_sensors"])
                    logger().debug(f"Received observation {step_observation}")

            # (3) Action selection (epsilon-greedy) and dispatch to controller.
            if not step_send_action:
                action = self.policy(step_observation)
                queue.send({"action": action})
                actions.append(action)
                step_send_action = True

            # (2) Blocking wait for end step controller message.
            if not step_control:
                step_messages = queue.search_message("step")
                if not step_messages:
                    continue
                else:
                    step_object = step_messages[0]
                    if step_object["step"] != self.environment.step_index:
                        raise RuntimeError(
                            f"Controller step index {step_object['step']} does not match "
                            f"supervisor step index {self.environment.step_index}."
                        )
                    else:
                        step_control = True

            # (4) Environment step: obtain new state and reward.
            state, reward = self.environment.step()
            rewards.append(reward)
            logger().debug(
                f"Step {self.environment.step_index + 1}: Distance to finish line: {state.finish_line_distance:.4f}"
            )

            # (5) Termination check: restart controller and exit loop if episode ends.
            if state.is_terminated:
                break

            self.environment.step_index += 1
            step_observation = None
            step_send_action = False
            step_control = False

        logger().info(f"Simulation terminated at step {state.step_index + 1}, success: {state.is_success}")
        return np.array(observations, dtype=int), np.array(actions, dtype=int), np.array(rewards, dtype=float)
