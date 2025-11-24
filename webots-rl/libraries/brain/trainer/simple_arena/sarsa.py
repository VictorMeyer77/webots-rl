"""
Trainer for tabular SARSA in the simple arena environment.

Message / supervisor loop protocol (per step):
  1. Sync: send {'sync': 1} until controller answers with 'ack'.
  2. Observation: wait for {'observation': {...}} containing distance sensors.
  3. Action: choose ε-greedy action and send {'action': int}.
  4. Step confirm: wait for {'step': step_index} to align controller & supervisor.
  5. Environment step: call environment.step() -> (StateSimpleArena, reward).
  6. Update: apply SARSA update using previous (state, action) and current (state, action).
  7. Termination: break when state.is_terminated.

Notes:
  * Observation discretization defined by OBSERVATION_SIZE * OBSERVATION_CARDINALITY.
  * Q-table stored in parent TrainerSarsa.
  * Reward shaping handled inside EnvironmentSimpleArena.step().
  * Current implementation uses previous state/action as "next" in update; ensure ordering consistency.
"""

import random

import numpy as np
from brain.environment import Environment
from brain.trainer.sarsa import TrainerSarsa
from brain.utils.logger import logger

OBSERVATION_SIZE = 8  # Number of distance sensors
OBSERVATION_CARDINALITY = 3  # Number of discrete bins per sensor
ACTION_SIZE = 3  # Number of discrete actions


class TrainerSarsaSimpleArena(TrainerSarsa):
    """
    SARSA trainer for the simple arena task.

    Responsibilities:
      * Implements ε-greedy policy over tabular Q-values.
      * Manages message handshake and step synchronization.
      * Executes a full episode and applies on-policy TD updates.

    Attributes:
      alpha (float): Learning rate for SARSA updates (inherited).
      gamma (float): Discount factor (inherited).
      epsilon (float): Current exploration rate (inherited).
    """

    alpha: float
    gamma: float
    epsilon: float

    def __init__(
        self,
        environment: Environment,
        model_name: str,
        alpha: float,
        gamma: float,
        epsilon: float,
    ):
        """
        Initialize SARSA trainer and allocate Q-table.

        Parameters:
          environment (Environment): Provides step() and queue messaging interface.
          model_name (str): Base name used when persisting the Q-table.
          alpha (float): Learning rate for temporal difference updates.
          gamma (float): Discount factor applied to future value estimates.
          epsilon (float): Initial ε for ε-greedy exploration policy.
        """
        super().__init__(
            environment=environment,
            model_name=model_name,
            action_size=ACTION_SIZE,
            observation_size=OBSERVATION_SIZE,
            observation_cardinality=OBSERVATION_CARDINALITY,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
        )

    def policy(self, observation: dict) -> int:
        """
        ε-greedy action selection.

        Chooses a random action with probability ε; otherwise selects an action
        among those with maximal Q-value (uniform tie-breaking).

        Parameters:
          observation (dict): Must include key 'distance_sensors' -> sequence of numeric values.

        Returns:
          int: Selected discrete action index.
        """
        observation = np.array(observation["distance_sensors"])
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

    def simulation(self) -> float:
        """
        Run one complete SARSA training episode.

        Loop details:
          * Synchronize with controller (ack handshake).
          * Receive observation -> select/send action -> confirm step alignment.
          * Perform environment step and accumulate reward.
          * After the first transition, invoke update_q_table each step using current
            and previous (observation, action) pairs. (Ordering currently reversed.)
          * Terminate when environment signals episode end.

        Returns:
          float: Total accumulated reward for the episode.
        """
        queue = self.environment.queue
        total_reward = 0.0
        previous_reward = 0.0
        state = None
        sync = False
        step_observation = None
        previous_step_observation = None
        step_action = None
        previous_step_action = None
        step_control = False

        # Main supervisor-driven loop; exits on Webots termination (-1) or episode end.
        while self.environment.supervisor.step(self.environment.timestep) != -1:

            queue.clear_buffer()

            # (1) Initial synchronization handshake on the very first step.
            if not sync:
                if not queue.search_message("ack"):
                    queue.send({"sync": 1})
                    logger().debug("Sent sync message to controller.")
                    continue
                else:
                    sync = True
                    logger().debug("Synchronization with controller successful.")

            # (2) Blocking wait for an observation message.
            if step_observation is None:
                observation_messages = queue.search_message("observation")
                if not observation_messages:
                    continue
                else:
                    step_observation = observation_messages[0]["observation"]
                    logger().debug(f"Received observation {step_observation}")

            # (3) Action selection (epsilon-greedy) and dispatch to controller.
            if step_action is None:
                step_action = self.policy(step_observation)
                queue.send({"action": step_action})

            # (4) Blocking wait for end step controller message.
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

            # (5) Environment step: obtain new state and reward, update q_table.
            state, reward = self.environment.step()
            total_reward += reward

            if previous_step_observation is not None:
                self.update_q_table(
                    np.array(previous_step_observation["distance_sensors"]),
                    previous_step_action,
                    previous_reward,
                    np.array(step_observation["distance_sensors"]),
                    step_action,
                )

            # (6) Termination check: update q_table and exit loop if episode ends.
            if state.is_terminated:
                self.update_q_table(
                    np.array(step_observation["distance_sensors"]), step_action, reward, None, None, terminated=True
                )
                break

            # (7) Set variables for next iteration.
            self.environment.step_index += 1
            previous_step_observation = step_observation
            previous_step_action = step_action
            previous_reward = reward
            step_observation = None
            step_action = None
            step_control = False
            logger().debug(f"Controller completed step {self.environment.step_index}")

        logger().info(f"Simulation terminated at step {state.step_index + 1}, success: {state.is_success}")
        return total_reward
