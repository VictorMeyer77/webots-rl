"""
REINFORCE Trainer for Simple Arena Navigation Task.

This module implements a REINFORCE policy gradient trainer specialized for the simple
arena environment in Webots. The trainer manages communication between a supervisor
(this script) and a robot controller using a message-passing queue protocol.

Environment:
    Simple arena with an e-puck robot navigating to a finish line using distance sensors.

    Observations:
        - 8 distance sensor readings (infrared proximity sensors)
        - Values typically in range [0, 1000+] (closer objects = higher values)

    Actions:
        - 0: Move forward
        - 1: Turn left
        - 2: Turn right

    Rewards:
        - Computed by environment based on proximity to finish line
        - May include penalties for collisions or time

Notes:
    - The controller must be running in parallel for message-passing to work
    - Episode length is bounded by environment.episode_size to prevent infinite loops
    - Observations are stored as raw distance sensor lists then converted to np.float32
    - Actions are discrete integers in range [0, ACTION_SIZE-1]
"""

import numpy as np
import tensorflow as tf
from brain.trainer.reinforce import TrainerReinforce
from brain.utils.logger import logger

ACTION_SIZE = 3  # Number of discrete actions (forward, left, right)


class TrainerReinforceSimpleArena(TrainerReinforce):
    """
    REINFORCE trainer for simple arena navigation using distance sensors.

    Concrete implementation of TrainerReinforce for the simple arena environment
    in Webots. This trainer handles the communication protocol between the supervisor
    (training script) and the robot controller, collects episode trajectories using
    distance sensor observations, and trains a policy network to navigate to the
    finish line.

    Observation Space:
        Dictionary with key "distance_sensors" containing a list of 8 float values
        representing infrared proximity sensor readings from the e-puck robot.
        Higher values indicate closer obstacles.

    Action Space:
        Discrete actions (integers 0-2):
        - 0: Move forward
        - 1: Turn left
        - 2: Turn right

    Methods:
        policy: Select action from distance sensor observation (stochastic)
        simulation: Run one episode and collect trajectory data
    """

    def policy(self, observation: dict) -> int:
        """
        Select action stochastically based on distance sensor observations.

        Implements a stochastic policy π(a|s) by computing action probabilities from
        the policy network and sampling from the resulting distribution. This ensures
        exploration during training, which is essential for REINFORCE to discover
        good behaviors.

        Args:
            observation (dict): Observation dictionary from the controller.
                Expected structure: {"distance_sensors": [v0, v1, ..., v7]}
                where each v_i is a float representing proximity sensor reading.
                Typical values: 0 (far) to 1000+ (very close)

        Returns:
            int: Selected action index in range [0, ACTION_SIZE-1].
                - 0: Move forward
                - 1: Turn left
                - 2: Turn right
        """
        observation = np.atleast_2d(np.array(observation["distance_sensors"]).astype(np.float32))
        logits = self.model(observation)[0]
        probs = tf.nn.softmax(logits).numpy()
        return np.random.choice(len(probs), p=probs)

    def simulation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run one complete episode and collect trajectory data via supervisor-controller protocol.

        Executes a full episode of interaction between the supervisor (this trainer) and
        the robot controller in Webots. Uses a message-passing queue to synchronize actions,
        observations, and step completion. Collects observations, actions, and rewards for
        the entire episode, which are later used to compute policy gradients.

        Communication Protocol:
            **Phase 1: Initial Synchronization**
            - Supervisor sends {"sync": 1} repeatedly
            - Waits for controller to respond with {"ack": 1}
            - This handshake ensures both processes are ready

            **Phase 2: Episode Loop** (repeated until termination)
            For each timestep:
                1. Wait for observation message from controller
                2. Extract observation: {"observation": {"distance_sensors": [...]}}
                3. Select action using self.policy()
                4. Send action to controller: {"action": action_id}
                5. Wait for step completion: {"step": step_index}
                6. Validate step indices match (supervisor vs controller)
                7. Compute reward via environment.step()
                8. Check termination condition
                9. Increment step_index and reset flags for next iteration

            **Phase 3: Termination**
            - Episode ends when state.is_terminated == True
            - Log final statistics (step count, success status)
            - Return collected trajectory arrays

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Episode trajectory data:
                - observations (np.ndarray): Distance sensor readings per timestep.
                    Shape: (episode_length, 8)
                    dtype: np.float32
                    Each row: 8 distance sensor values

                - actions (np.ndarray): Actions taken per timestep.
                    Shape: (episode_length,)
                    dtype: np.int64
                    Values: 0 (forward), 1 (left), 2 (right)

                - rewards (np.ndarray): Immediate rewards per timestep.
                    Shape: (episode_length,)
                    dtype: np.float32
                    Values: Computed by environment reward function
        """

        observations = []
        actions = []
        rewards = []
        queue = self.environment.queue
        state = None
        sync = False
        step_observation = None
        step_action = None
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
                    observations.append(step_observation["distance_sensors"])
                    logger().debug(f"Received observation {step_observation}")

            # (3) Action selection (epsilon-greedy), dispatch to controller and set in environment to reward compute.
            if step_action is None:
                step_action = self.policy(step_observation)
                queue.send({"action": step_action})
                actions.append(step_action)
                self.environment.last_action = step_action

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
            step_action = None
            step_control = False

        logger().info(f"Simulation terminated at step {state.step_index + 1}, success: {state.is_success}")
        return (
            np.array(observations, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
        )
