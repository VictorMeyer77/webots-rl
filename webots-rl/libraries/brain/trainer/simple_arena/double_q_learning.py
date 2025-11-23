"""
Trainer for tabular Double Q-learning in the simple arena environment.

Message / supervisor loop protocol (per step):
  1. Sync: send {'sync': 1} until controller answers with 'ack'.
  2. Observation: wait for {'observation': {...}} containing distance sensors.
  3. Action: choose ε-greedy action and send {'action': int}.
  4. Step confirm: wait for {'step': step_index} to align controller & supervisor.
  5. Environment step: call environment.step() -> (StateSimpleArena, reward).
  6. Update: apply Double Q-learning update using previous (state, action, reward)
     and current next_state, updating either Q_A or Q_B.
  7. Termination: break when state.is_terminated.

Notes:
  * Observation discretization defined by OBSERVATION_SIZE * OBSERVATION_CARDINALITY.
  * Two Q-tables (Q_A and Q_B) are stored in the parent TrainerDoubleQLearning.
  * Reward shaping handled inside EnvironmentSimpleArena.step().
  * The implementation updates either Q_A(s_t, a_t) or Q_B(s_t, a_t) using r_t
    and the value from the opposite table at the next greedy action, consistent
    with Double Q-learning.
"""

import random

import numpy as np
from brain.environment import Environment
from brain.trainer.double_q_learning import TrainerDoubleQLearning
from brain.utils.logger import logger

OBSERVATION_SIZE = 8  # Number of distance sensors
OBSERVATION_CARDINALITY = 3  # Number of discrete bins per sensor
ACTION_SIZE = 3  # Number of discrete actions


class TrainerDoubleQLearningSimpleArena(TrainerDoubleQLearning):
    """Double Q-learning trainer for the simple arena task.

    Responsibilities:
      * Implements an ε-greedy policy over tabular Q-values derived from
        two Q-tables (Q_A and Q_B).
      * Manages message handshake and step synchronization with the controller.
      * Executes a full episode and applies Double Q-learning updates
        (updating either Q_A or Q_B at each step).

    Attributes:
      alpha (float): Learning rate for Double Q-learning updates (inherited).
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

    def policy(self, observation: dict, table: str) -> int:
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
            q_values = (
                self.model_a.q_table[self.model_a.observation_to_index(observation)]
                if table == "A"
                else self.model_b.q_table[self.model_b.observation_to_index(observation)]
            )
            max_q = q_values.max()
            max_indexes = np.where(q_values == max_q)[0]
            action = np.random.choice(max_indexes)
            logger().debug(f"Taking best action {action} for state {observation} with Q-value {max_q}")
        return int(action)

    def simulation(self) -> float:
        """Run one complete Double Q-learning training episode.

        Loop details:
          * Synchronize with controller (ack handshake).
          * Receive observation -> select/send action -> confirm step alignment.
          * Perform environment step and accumulate reward.
          * After the first transition, invoke ``update_q_table`` each step
            using the previous (observation, action, reward) triple and the
            current observation as the next state. On each step, randomly
            choose which Q-table ("A" or "B") to update, following the
            Double Q-learning rule.

          * On terminal transition, update the last (state, action) pair
            in the chosen table with target equal to the final reward
            (no bootstrap term), then exit.

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
            # Randomly choose which Q-table ('A' or 'B') to use/update on this step (Double Q-learning).
            q_action_table = "A" if random.random() < 0.5 else "B"

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
                    logger().debug(f"Received observation {step_observation}")

            # (3) Action selection (epsilon-greedy) and dispatch to controller.
            if step_action is None:
                step_action = self.policy(step_observation, q_action_table)
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
                    q_action_table,
                )

            # (6) Termination check: update q_table and exit loop if episode ends.
            if state.is_terminated:
                self.update_q_table(
                    np.array(step_observation["distance_sensors"]),
                    step_action,
                    reward,
                    None,
                    q_action_table,
                    terminated=True,
                )
                break

            # (7) Set variables for next iteration.
            self.environment.step_index += 1
            previous_step_observation = step_observation
            previous_reward = reward
            previous_step_action = step_action
            step_observation = None
            step_action = None
            step_control = False
            logger().debug(f"Controller completed step {self.environment.step_index}")

        logger().info(f"Simulation terminated at step {state.step_index + 1}, success: {state.is_success}")
        return total_reward
