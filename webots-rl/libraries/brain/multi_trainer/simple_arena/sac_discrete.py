import json
from collections import deque

import brain.utils.image as img
import brain.utils.tcp_socket as tcp
import numpy as np
from brain.multi_trainer.agent import MultiTrainerAgent
from brain.utils.logger import logger


class TrainerAgentSACDiscreteSimpleArena(MultiTrainerAgent):

    def tcp_send_step(
        self, observation: np.ndarray, action: int, reward: float, done: bool, next_observation: np.ndarray
    ) -> None:
        message = json.dumps(
            {
                "message_type": "step",
                "observation": observation.tolist(),
                "action": action,
                "reward": reward,
                "done": done,
                "next_observation": next_observation.tolist(),
            }
        )
        tcp.send(self.connection, message)

    def simulation(self) -> float:

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

            if previous_step_observation is not None:
                self.tcp_send_step(
                    observation=previous_step_observation,
                    action=previous_step_action,
                    reward=previous_reward,
                    done=False,
                    next_observation=step_observation,
                )
            if state.is_terminated:
                self.tcp_send_step(
                    observation=step_observation,
                    action=step_action,
                    reward=reward,
                    done=True,
                    next_observation=step_observation,  # unused
                )
                break

            # (8) Set variables for next iteration.
            self.environment.step_index += 1

            previous_step_observation = step_observation
            step_observation = None
            previous_step_action = step_action
            step_action = None
            previous_reward = reward
            step_control = False
            logger().debug(f"Controller completed step {self.environment.step_index}")

        logger().info(f"Simulation terminated at step {state.step_index + 1}, success: {state.is_success}")
        return total_reward
