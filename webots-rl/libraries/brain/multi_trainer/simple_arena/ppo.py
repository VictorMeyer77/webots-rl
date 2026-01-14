"""
PPO (Proximal Policy Optimization) agent for Simple Arena multi-environment training.

This module implements a PPO agent specifically designed for the Simple Arena
environment. It handles communication between the Webots supervisor and the PPO
trainer via TCP sockets, manages frame stacking for visual observations, and
coordinates the episode execution flow.

The agent uses a synchronization protocol to ensure proper coordination between
the supervisor-driven simulation loop and the robot controller, collecting
experience data (observations, actions, rewards) and sending it to the trainer
for PPO updates.
"""

import json
from collections import deque

import brain.utils.image as img
import brain.utils.tcp_socket as tcp
import numpy as np
from brain.multi_trainer.agent import MultiTrainerAgent
from brain.utils.logger import logger


class TrainerAgentPPOSimpleArena(MultiTrainerAgent):
    """
    PPO agent for Simple Arena multi-environment reinforcement learning.

    This class implements the agent-side logic for PPO training in the Simple
    Arena environment. It runs within the Webots supervisor process and handles:
    - Synchronization with the robot controller
    - Visual observation processing and frame stacking
    - Communication with the PPO trainer via TCP
    - Episode management and termination detection

    The agent operates in a supervisor-driven loop where the supervisor controls
    the simulation timestep. It exchanges messages with the controller via a
    message queue and with the trainer via TCP sockets. The controller provides
    camera observations and receives motor commands, while the trainer provides
    action selections based on the current policy.
    """

    def simulation(self) -> float:
        """
        Executes a complete training episode in the Simple Arena environment.

        This method implements the main episode loop that coordinates between the
        Webots supervisor, the robot controller, and the PPO trainer. The loop
        follows a structured protocol to ensure proper synchronization:

        1. Initial Synchronization: Performs a handshake with the controller to
           establish communication before starting the episode. Sends "sync"
           messages until receiving an "ack" response.

        2. Observation Processing: Waits for camera observations from the
           controller, processes them (grayscale conversion, resizing, normalization),
           stacks 4 consecutive frames for temporal information, and sends them to
           the trainer via TCP.

        3. Action Selection: Receives action and value estimates from the trainer
           via TCP and forwards the action to the controller for execution.

        4. Step Synchronization: Waits for step completion confirmation from the
           controller to ensure the supervisor and controller remain synchronized.

        5. Environment Update: Advances the environment state, computes rewards,
           checks for episode termination, and sends experience tuples
           (observation, action, reward, done, value) to the trainer.

        6. Iteration Management: Resets step variables and increments the step
           counter for the next iteration.

        The loop exits when either:
        - The episode terminates (success or failure condition met)
        - Webots signals shutdown (supervisor.step returns -1)

        Frame stacking is used to provide temporal context to the policy network,
        allowing it to infer motion and velocity information from static images.
        A deque with maxlen=4 automatically maintains the 4 most recent frames.

        Returns:
            float: Total cumulative reward obtained during the episode. This value
                is used for logging and monitoring training progress.

        Note:
            The method uses blocking waits for synchronization points (observation
            messages, action from trainer, step confirmation) to ensure proper
            coordination. This prevents race conditions and maintains consistent
            timing between components.
        """
        queue = self.environment.queue
        total_reward = 0.0
        state = None
        sync = False
        step_observation = None
        step_action = None
        step_value = None
        step_control = False
        frames = deque(maxlen=4)

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

            # (2) Blocking wait for an observation message. Process camera image.
            if step_observation is None:
                observation_messages = queue.search_message("observation")
                if not observation_messages:
                    continue
                else:
                    step_observation = observation_messages[0]["observation"]
                    frame = np.array(step_observation["camera"]).astype(np.uint8)
                    frame = img.format_image(frame, shape=(42, 42), grayscale=True, normalize=True)
                    frames.append(frame)
                    step_observation = img.concatenate_frames(frames, 4)
                    self.tcp_send_observation(step_observation)
                    logger().debug("Received camera image.")

            # (3) Action selection (epsilon-greedy) and dispatch to controller.
            if step_action is None:
                tcp_message = tcp.read(self.connection)
                if tcp_message is not None:
                    tcp_message = json.loads(tcp_message)
                    step_action = tcp_message["action"]
                    step_value = tcp_message["value"]
                    queue.send({"action": step_action})
                    self.environment.last_action = step_action

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

            # (6) Termination check: update q_table and exit loop if episode ends.

            self.tcp_send_step(
                observation=step_observation,
                action=step_action,
                reward=reward,
                done=state.is_terminated,
                value=step_value,
            )
            if state.is_terminated:
                break

            # (8) Set variables for next iteration.
            self.environment.step_index += 1

            step_observation = None
            step_action = None
            step_value = None
            step_control = False
            logger().debug(f"Controller completed step {self.environment.step_index}")

        logger().info(f"Simulation terminated at step {state.step_index + 1}, success: {state.is_success}")
        return total_reward
