"""
A2C agent for the Simple Arena environment.

This module implements an Advantage Actor-Critic (A2C) agent specifically designed
for the Simple Arena robot navigation task in Webots. The agent processes camera
images, maintains frame stacking for temporal awareness, and coordinates communication
between the Webots controller and the centralized A2C trainer.

Architecture:
  * Environment: Webots simulation with a robot equipped with a camera sensor.
  * Controller: Low-level robot controller running in Webots (handles motors, sensors).
  * Agent (this class): High-level decision-making layer that processes observations
    and communicates with the trainer.
  * Trainer: Centralized A2C trainer managing policy updates across multiple agents.

Notes:
  * The controller must implement the message protocol (sync, observation, action, step).
  * Frame stacking requires 4 consecutive frames before the first action.
  * The agent blocks waiting for messages but allows Webots simulation to continue.
  * Reward computation is environment-specific (defined in self.environment.step()).
"""

from collections import deque

import brain.utils.image as img
import brain.utils.tcp_socket as tcp
import numpy as np
from brain.multi_trainer.agent import MultiTrainerAgent
from brain.utils.logger import logger


class TrainerAgentA2CSimpleArena(MultiTrainerAgent):
    """
    A2C agent for robot navigation in the Simple Arena environment.

    This agent implements the client-side logic for an Advantage Actor-Critic
    system in a Webots robot navigation task. It handles camera image processing,
    frame stacking for temporal awareness, and bidirectional communication with
    both the Webots controller (via message queue) and the centralized trainer
    (via TCP).

    The agent operates in a supervisor-driven loop where Webots controls the
    simulation timestep. At each step, the agent coordinates with the controller
    to receive observations, requests actions from the trainer, and sends
    experiences back for policy learning.

    Communication Protocol:
      * With Controller (Message Queue):
        - "sync" → "ack": Initial handshake
        - "observation": Camera data and sensor readings
        - "action": Motor commands
        - "step": Confirmation of action execution

      * With Trainer (TCP):
        - "policy" request: Send stacked frames, receive action + value
        - "step" message: Send transition (s, a, r, done, V)
        - "terminated": Signal episode completion

    Attributes:
        Inherited from MultiTrainerAgent:
        - environment: Webots environment wrapper
        - connection: TCP socket to trainer
        - tcp_port: Port for trainer communication
    """

    def simulation(self) -> float:
        """
        Execute one complete episode in the Simple Arena environment.

        Orchestrates the full lifecycle of an episode, from initial synchronization
        with the controller through continuous observation-action-reward cycles
        until episode termination. The method implements a state machine with
        multiple blocking waits to ensure proper synchronization between the
        supervisor, controller, and trainer.

        Episode Flow:
          1. **Synchronization Phase**:
             - Send "sync" message to controller until "ack" received.
             - Ensures controller is ready before starting episode.

          2. **Observation Phase**:
             - Wait for "observation" message containing camera data.
             - Process raw image: resize, grayscale, normalize.
             - Append to frame buffer (deque with maxlen=4).
             - Stack frames to create temporal observation.
             - Send stacked frames to trainer via TCP.

          3. **Action Selection Phase**:
             - Wait for trainer's response containing action and value.
             - Parse TCP message to extract action index and V(s).
             - Send action to controller via message queue.
             - Store action for experience replay.

          4. **Execution Phase**:
             - Wait for controller's "step" confirmation message.
             - Verify step index consistency between supervisor and controller.
             - Ensures action execution completed before proceeding.

          5. **Evaluation Phase**:
             - Call environment.step() to compute reward and next state.
             - Accumulate reward for episode return.
             - Send transition (s, a, r, done, V) to trainer.

          6. **Termination Check**:
             - Exit loop if state.is_terminated is True.
             - Otherwise, reset step variables and continue.

        State Variables (reset each iteration):
          * step_observation (np.ndarray | None): Processed stacked frames [42, 42, 4].
          * step_action (int | None): Action index selected by policy.
          * step_value (float | None): Critic's value estimate V(s).
          * step_control (bool): Whether controller confirmed action execution.

        Blocking Behavior:
          * The method blocks at multiple points waiting for messages.
          * supervisor.step() advances simulation but returns immediately if no message.
          * Each blocking wait uses continue to retry on the next timestep.
          * This ensures temporal alignment without busy-waiting.

        Returns:
            float: Total cumulative reward for the episode. Sum of all immediate
                rewards r_t received at each timestep: R = Σ r_t. Typical range
                depends on task (e.g., [0, 500] for successful navigation).

        Raises:
            RuntimeError: If controller's step index doesn't match supervisor's.
                This indicates a synchronization error in the message protocol.
            ConnectionError: If TCP connection to trainer is lost during episode.
            json.JSONDecodeError: If trainer sends malformed JSON response.

        Side Effects:
            * Advances Webots simulation via supervisor.step().
            * Sends messages to controller (sync, action) via message queue.
            * Sends messages to trainer (observation, step) via TCP.
            * Updates environment.last_action and environment.step_index.
            * Modifies frames deque (append processed images).

        Notes:
            * The first 3 steps only collect frames without taking actions
              (frame buffer needs 4 frames before stacking).
            * Rewards are scaled by REWARD_SCALE_FACTOR (100.0) before sending
              to trainer for numerical stability.
            * Episode terminates on success (goal reached), failure (collision),
              or timeout (max steps exceeded).
            * The supervisor.step() call returns -1 when Webots simulation ends
              (user closes simulation or supervisor terminates).
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
                reward=reward * 100.0,
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
