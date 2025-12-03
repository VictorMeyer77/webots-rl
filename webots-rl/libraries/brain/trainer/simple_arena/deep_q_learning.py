"""
Deep Q-Learning Trainer for Simple Arena Environment with Vision-Based Control.

This module implements a concrete Deep Q-Learning trainer for the simple arena robotic
navigation task using camera input. The agent (e-puck robot) learns to navigate from
a starting position to a finish line while avoiding obstacles, using only visual input
from an onboard camera.
"""

import random
from collections import deque

import brain.utils.image as img
import numpy as np
import tensorflow as tf
from brain.environment import Environment
from brain.trainer.deep_q_learning import TrainerDeepQLearning
from brain.utils.logger import logger

ACTION_SIZE = 4


class TrainerDeepQLearningSimpleArena(TrainerDeepQLearning):

    def __init__(
        self,
        environment: Environment,
        model_name: str,
        model: tf.keras.models.Sequential,
        memory_size: int,
        gamma: float,
        epsilon: float,
        epsilon_decay: float,
        batch_size: int,
        fit_frequency: int,
        update_target_model_frequency: int,
        per_alpha: float,
        per_beta_start: float,
    ):
        """
        Initialize Deep Q-Learning trainer for vision-based simple arena navigation.

        Sets up the trainer for an e-puck robot learning to navigate to a finish line
        using only camera input. Configures the CNN model for processing stacked frames,
        prioritized experience replay buffer, and all training hyperparameters.

        Args:
            environment (Environment): Simple arena environment instance with Webots supervisor.
                Must provide:
                - queue: Message queue for supervisor-controller communication
                - supervisor: Webots supervisor node for environment control
                - step(): Method returning (state, reward) tuple
            model_name (str): Base name for saving model checkpoints.
                Example: "simple_arena_dqn_vision"
                Saved to: MODEL_PATH/{model_name}.keras
            model (tf.keras.models.Sequential): Convolutional neural network for Q-value estimation.
                **Required architecture**:
                - Input shape: (42, 42, 4) - stacked grayscale frames
                - Output shape: (4,) - Q-values for 4 actions
                - Must be compiled with optimizer (e.g., Adam) and loss (e.g., Huber, MSE)
            memory_size (int): Maximum capacity of prioritized replay buffer.
                Recommended: 50,000 for simple arena (balance memory vs diversity)
                Higher values provide more diverse experiences but use more RAM (~350MB for 50k)
            gamma (float): Discount factor for future rewards.
                Recommended: 0.99 (robot navigation benefits from long-term planning)
                Range: [0, 1] where higher values prioritize future rewards
            epsilon (float): Initial exploration rate for epsilon-greedy policy.
                Recommended: 1.0 (start with 100% random exploration)
                The agent explores randomly to discover the finish line initially
            epsilon_decay (float): Multiplicative epsilon decay per epoch.
                Recommended: 0.995 (reaches ~0.01 after ~900 episodes)
                Controls how quickly the agent transitions from exploration to exploitation
            batch_size (int): Number of experiences sampled per training step.
                Recommended: 64 for vision-based learning
                Larger batches (128) = more stable but slower; smaller (32) = faster but noisier
            fit_frequency (int): Train model every N environment steps.
                Recommended: 4 (train after collecting 4 new experiences)
                Lower values (1) = more frequent training but slower episodes
            update_target_model_frequency (int): Update target network every N steps.
                Recommended: 1000 for simple arena
                Balances stability (higher values) vs adaptation speed (lower values)
            per_alpha (float): Prioritization exponent controlling sampling bias.
                Recommended: 0.6 (balance between uniform and full prioritization)
                Range: [0, 1] where 0=uniform, 1=full prioritization by TD-error
            per_beta_start (float): Initial importance sampling weight exponent.
                Recommended: 0.4 (annealed to 1.0 during training)
                Corrects bias from non-uniform sampling; full correction at end of training
        """
        super().__init__(
            environment=environment,
            model_name=model_name,
            model=model,
            memory_size=memory_size,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size,
            fit_frequency=fit_frequency,
            update_target_model_frequency=update_target_model_frequency,
            per_alpha=per_alpha,
            per_beta_start=per_beta_start,
        )

    def policy(self, observation: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy for vision-based navigation.

        Implements the epsilon-greedy exploration strategy that balances exploration
        (random actions) and exploitation (greedy actions based on learned Q-values).
        This is the core decision-making function during training episodes.

        Algorithm:
            1. Generate random number in [0, 1)
            2. If random < epsilon: Select random action (exploration)
            3. Else: Use CNN to predict Q-values and select argmax (exploitation)

        Exploration vs Exploitation:
            - **Exploration** (epsilon probability): Random action selection allows the
              agent to discover new states and potentially better strategies.
            - **Exploitation** (1-epsilon probability): Use learned policy to maximize
              expected reward based on current knowledge.

        Epsilon Annealing:
            - Starts high (typically 1.0 = 100% random) for initial exploration
            - Decays multiplicatively each epoch (e.g., epsilon *= 0.995)
            - Reaches minimum (typically 0.01 = 1% random) for stable exploitation
            - This ensures discovery early and refinement later in training

        Args:
            observation (np.ndarray): Preprocessed stacked frames representing current state.
                Shape: (42, 42, 4) where:
                - 42x42: Resized grayscale image dimensions
                - 4: Number of consecutive frames stacked for temporal context
                Values: [0.0, 1.0] (normalized pixel intensities)

        Returns:
            int: Selected action index in range [0, ACTION_SIZE-1].
                Action mapping:
                - 0: Move forward
                - 1: Turn left
                - 2: Turn right
                - 3: Move backward

        Behavior Details:
            **Random Action (exploration)**:
            - Uniformly samples from [0, 1, 2, 3]
            - Logged at DEBUG level with message "Taking random action {action}"
            - Critical for discovering finish line location early in training

            **Greedy Action (exploitation)**:
            - Adds batch dimension to observation: (42, 42, 4) → (1, 42, 42, 4)
            - Performs CNN forward pass to predict Q-values for all 4 actions
            - Selects action with maximum Q-value (argmax)
            - Logged at DEBUG level with message "Taking best action {action}"
            - Uses learned policy to navigate efficiently toward goal
        """
        if random.random() < self.epsilon:
            action = random.randint(0, ACTION_SIZE - 1)
            logger().debug(f"Taking random action {action}")
        else:
            observation = np.expand_dims(observation, axis=0)
            action_values = self.model.predict(observation, verbose=0)
            action = np.argmax(action_values)
            logger().debug(f"Taking best action {action}")
        return int(action)

    def simulation(self) -> float:
        """
        Execute one complete training episode in the simple arena environment.

        Runs a full episode of the e-puck robot navigating to the finish line using
        vision-based Deep Q-Learning. Handles all aspects of the training loop including
        supervisor-controller communication, camera image processing, frame stacking,
        action selection, experience storage, and periodic model training.

        Episode Flow (per step):
            1. **Synchronization**: Initial handshake with controller
            2. **Observation**: Receive camera image from controller
            3. **Image Processing**: Resize → grayscale → normalize → stack frames
            4. **Action Selection**: Epsilon-greedy policy using CNN or random
            5. **Action Execution**: Send action to controller via message queue
            6. **Step Confirmation**: Wait for controller to confirm step completion
            7. **Reward Computation**: Environment computes reward and checks termination
            8. **Experience Storage**: Add (s, a, r, s', done) to prioritized replay buffer
            9. **Model Training**: Fit model every fit_frequency steps
            10. **Target Update**: Sync target network every update_target_model_frequency steps
            11. **Episode End**: Break on termination (success, collision, timeout)

        Message Protocol (Supervisor ↔ Controller):
            The supervisor and controller communicate via a message queue with strict synchronization:

            **Initial Sync (once per episode)**:
            - Supervisor → Controller: {"sync": 1}
            - Controller → Supervisor: {"ack": 1}
            - Purpose: Ensure both are ready before starting episode

            **Per-Step Exchange (repeated until episode end)**:
            - Controller → Supervisor: {"observation": {"camera": [...], "distance_sensors": [...]}}
            - Supervisor → Controller: {"action": int}
            - Controller → Supervisor: {"step": int}
            - Purpose: Exchange observation for action, confirm step completion

        Vision Processing Pipeline:
            Raw camera → Processed frame → Stacked frames → Network input

            1. Raw RGB image: (52, 39, 3) uint8 [0-255]
            2. Resize: (42, 42, 3)
            3. Grayscale: (42, 42) uint8 [0-255]
            4. Normalize: (42, 42) float32 [0.0-1.0]
            5. Stack 4 frames: (42, 42, 4) - temporal context
            6. Feed to CNN for Q-value prediction

        Frame Stacking Details:
            - Maintains deque with maxlen=4 for automatic old frame removal
            - Each step appends new processed frame
            - Concatenates frames along channel dimension: (42, 42, 4)
            - Provides motion/velocity information to the network
            - **Critical**: Frame buffer reset at episode start to avoid temporal leakage

        Experience Storage Strategy:
            - Stores transitions in prioritized replay buffer with max priority
            - Non-terminal transitions: (s_t, a_t, r_t, s_t+1, False)
            - Terminal transition: (s_T, a_T, r_T, s_T, True)
            - Note: Terminal next_state is same as current state (unused)

        Training Triggers:
            - **Model Training**: Every fit_frequency steps (e.g., every 4 steps)
              - Only if buffer has >= batch_size experiences
              - Samples prioritized batch and updates Q-network
            - **Target Update**: Every update_target_model_frequency steps (e.g., every 1000 steps)
              - Copies main network weights to target network
              - Provides stable Q-value targets

        Episode Termination Conditions:
            - **Success**: Robot reaches finish line (state.is_success = True)
            - **Timeout**: Maximum steps exceeded
            - **Simulation Error**: Webots supervisor.step() returns -1

        Variable State Tracking:
            - step_observation: Current processed stacked frames (42, 42, 4)
            - previous_step_observation: Previous stacked frames for experience tuple
            - step_action: Current action selected by policy
            - previous_step_action: Previous action for experience tuple
            - step_control: Flag indicating controller confirmed step completion
            - sync: Flag indicating initial synchronization completed
            - frames: Deque of 4 most recent processed camera frames

        Returns:
            float: Total cumulative reward for the episode.
                Sum of all rewards received from environment.step() calls.
                Logged to TensorBoard for tracking training progress.
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
                    logger().debug("Received camera image.")

            # (3) Action selection (epsilon-greedy) and dispatch to controller.
            if step_action is None:
                step_action = self.policy(step_observation)
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
                self.memory.add(
                    previous_step_observation,
                    previous_step_action,
                    previous_reward,
                    step_observation,
                    False,
                )

            if state.is_terminated:
                self.memory.add(
                    step_observation,
                    step_action,
                    reward,
                    step_observation,  # unsued
                    True,
                )
                break

            # (7) Periodic model training and target network update.
            if len(self.memory) >= self.batch_size and self.environment.step_index % self.fit_frequency == 0:
                self.fit_model()

            if self.environment.step_index % self.update_target_model_frequency == 0:
                self.update_target_model()

            # (8) Set variables for next iteration.
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
