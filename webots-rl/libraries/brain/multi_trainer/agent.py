"""
Multi-environment agent module for distributed reinforcement learning.

This module provides an abstract base class for agents that interact with
simulation environments and communicate with a centralized trainer via TCP.
Each agent runs in its own process, executing episodes and sending experiences
to the trainer for policy updates.

Architecture:
  * Agent-Trainer Communication: Agents act as TCP clients connecting to the
    trainer server. They request actions and send transition data asynchronously.
  * Distributed Execution: Multiple agents run in parallel, each managing its
    own environment instance and communicating with the shared trainer.
  * Message Protocol: JSON-based protocol with three message types:
      - "policy": Request action for current observation.
      - "step": Send transition (s, a, r, done, V) to trainer's memory.
      - "terminated": Signal completion and disconnect.

Workflow (per agent):
  1. Initialization: Connect to trainer's TCP server on specified port.
  2. Episode Loop: For each epoch:
     a. Reset environment to initial state.
     b. Execute simulation() to run one episode.
     c. Send observations and receive actions from trainer.
     d. Store transitions via tcp_send_step().
     e. Log episode reward.
  3. Termination: Send "terminated" message and close connection.

Message Protocol:
  * Policy Request (agent → trainer):
      {"message_type": "policy", "observation": [[...]]}
    Trainer Response:
      {"action": 2, "value": 0.45}

  * Step Message (agent → trainer):
      {
        "message_type": "step",
        "observation": [[...]],
        "action": 2,
        "reward": 1.0,
        "done": false,
        "value": 0.45
      }

  * Termination (agent → trainer):
      {"message_type": "terminated"}

Key Features:
  * Automatic reconnection attempts (10 tries with 1-second delays).
  * Non-blocking socket reads with configurable timeout.
  * Abstract simulation() method for custom environment logic.
  * Graceful shutdown protocol with trainer acknowledgment.

Notes:
  * The simulation() method must be implemented by subclasses to define
    environment-specific behavior (observation processing, action execution).
  * All agents share the same policy parameters via the centralized trainer.
  * Socket timeout prevents blocking when no data is available from trainer.
  * The agent expects the trainer to be running before initialization.
"""

import json
import socket
import time
from abc import ABC, abstractmethod

import brain.utils.tcp_socket as tcp
import numpy as np
from brain.environment import Environment
from brain.utils.logger import logger

TCP_HOST = "127.0.0.1"
SOCKET_TIMEOUT = 0.001


class MultiTrainerAgent(ABC):
    """
    Abstract base class for agents in distributed reinforcement learning setups.

    This class manages the client-side logic for agents that interact with
    simulation environments and communicate with a centralized trainer via TCP.
    Each agent runs independently, executing episodes and sending transition
    data to the trainer for policy learning.

    The agent operates as a TCP client, connecting to the trainer server and
    exchanging JSON messages according to a defined protocol. The abstract
    simulation() method must be implemented by subclasses to define the
    environment-specific interaction logic.

    Attributes:
        environment (Environment): Simulation environment managed by this agent.
            Should implement reset(), step(), and observation methods.
        connection (socket.socket | None): TCP socket connection to the trainer
            server. None if not yet connected.
        tcp_port (int): Port number for connecting to the trainer server.
            Each agent typically uses a unique port.
    """

    environment: Environment
    connection: socket.socket | None = None
    tcp_port: int

    def __init__(self, environment: Environment, tcp_port: int):
        """
        Initialize the multi-trainer agent with an environment and TCP port.

        Establishes the simulation environment and initiates connection to the
        trainer server. The connection process includes automatic retry logic
        to handle cases where the trainer hasn't started yet.

        Args:
            environment (Environment): The simulation environment instance that
                this agent will interact with. Must provide methods for reset(),
                step(action), and getting observations.
            tcp_port (int): TCP port number for connecting to the trainer server.
                Should match the port assigned to this agent by the trainer.
                Common pattern: base_port + agent_index.
        """
        self.environment = environment
        self.tcp_port = tcp_port
        self._init_client()

    def _init_client(self):
        """
        Initialize TCP client connection to the trainer server with retry logic.

        Attempts to establish a socket connection to the trainer server up to
        10 times with 1-second delays between attempts. This retry mechanism
        allows agents to start before the trainer is ready, improving robustness
        in distributed setups.

        Once connected, sets the socket timeout to SOCKET_TIMEOUT (0.001s) to
        enable non-blocking reads during the training loop.

        Connection Parameters:
            * Host: TCP_HOST (127.0.0.1 for local training)
            * Port: self.tcp_port (assigned during initialization)
            * Timeout: SOCKET_TIMEOUT (1ms) for non-blocking operations

        Raises:
            ConnectionError: If connection fails after 10 retry attempts.

        Side Effects:
            * Sets self.connection to a connected socket instance.
            * Logs connection status messages via logger.

        Notes:
            * The 1-second delay between retries prevents busy-waiting.
            * The short timeout (1ms) allows the agent to poll frequently
              without blocking the simulation loop.
            * IPv4 (AF_INET) and TCP (SOCK_STREAM) are used for reliability.
        """
        for _ in range(10):
            try:
                self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.connection.connect((TCP_HOST, self.tcp_port))
                self.connection.settimeout(SOCKET_TIMEOUT)
                logger().info("TCP client: connected with server")
                return
            except ConnectionRefusedError:
                logger().info("TCP client: waiting for TCP server...")
                time.sleep(1)
        raise ConnectionError(f"Could not connect to TCP server ({TCP_HOST}:{self.tcp_port}) after multiple attempts")

    @abstractmethod
    def simulation(self) -> float:
        """
        Execute one complete episode in the environment.

        This abstract method must be implemented by subclasses to define the
        agent's behavior during a single episode. Implementations should:
          1. Get initial observation from environment.
          2. Loop until episode terminates:
             a. Request action from trainer via tcp_send_observation().
             b. Execute action in environment.
             c. Receive reward and next observation.
             d. Send transition to trainer via tcp_send_step().
          3. Return total episode reward.

        The method encapsulates the core interaction loop between agent and
        environment, coordinating with the trainer to receive actions based
        on the current policy.

        Returns:
            float: Total cumulative reward obtained during the episode.
                This is the sum of all immediate rewards r_t received at
                each timestep: R = Σ r_t.
        """
        raise NotImplementedError("simulation() method must be implemented by subclass")

    def run(self, epochs: int) -> None:
        """
        Execute the main training loop for a specified number of epochs.

        Runs multiple episodes (epochs) in sequence, coordinating with the trainer
        to receive policy updates between episodes. After each episode, resets
        the environment and logs the performance. Upon completion, notifies the
        trainer that this agent is terminating.

        Training Loop Structure:
          1. For each epoch (episode):
             a. Execute simulation() to run one complete episode.
             b. Reset environment to initial state.
             c. Log episode reward and progress.
          2. Send termination message to trainer.
          3. Agent is ready to disconnect.

        Args:
            epochs (int): Number of episodes (epochs) to execute. Each epoch
                corresponds to one complete trajectory from initial state to
                terminal state. Common values: 100-10000 depending on task
                complexity and training objectives.

        Side Effects:
            * Calls simulation() epochs times, triggering TCP communication.
            * Resets environment after each episode.
            * Logs progress messages via logger.
            * Sends "terminated" message to trainer at completion.

        Notes:
            * The environment's reset() is called after each episode to prepare
              for the next trajectory.
            * Epoch indexing in logs is 1-based for readability (1/100, 2/100, ...).
            * The trainer may continue running after this agent terminates if
              other agents are still active.
        """
        for epoch in range(epochs):

            reward = self.simulation()
            self.environment.reset()
            logger().info(f"Epoch {epoch + 1}/{epochs} completed with reward {reward}")

        self.tcp_send_terminated()

    def tcp_send_observation(self, observation: np.ndarray) -> None:
        """
        Send observation to trainer and request an action.

        Packages the current observation into a JSON "policy" message and sends
        it to the trainer. The trainer will respond with an action selected by
        the current policy and a value estimate for the state.

        This method implements the request part of the request-response protocol
        for action selection. After calling this, the agent should read the
        trainer's response to obtain the action.

        Args:
            observation (np.ndarray): Current state observation from the environment.
                Shape depends on the environment (e.g., [84, 84, 4] for stacked
                frames or [features] for vector observations). Will be serialized
                to JSON via tolist().

        Side Effects:
            * Sends JSON message over TCP connection.
            * Does not wait for response (non-blocking send).

        Message Format:
            {"message_type": "policy", "observation": [[...]]}

        Notes:
            * The observation array is converted to nested Python lists via tolist()
              for JSON serialization.
            * Large observations (e.g., high-resolution images) may introduce
              network latency.
            * The trainer's response includes both action and value for efficiency.
        """
        message = json.dumps({"message_type": "policy", "observation": observation.tolist()})
        tcp.send(self.connection, message)

    def tcp_send_step(self, observation: np.ndarray, action: int, reward: float, done: bool, value: float) -> None:
        """
        Send transition tuple to trainer for storage in experience memory.

        Packages a complete transition (s, a, r, done, V(s)) into a JSON "step"
        message and sends it to the trainer. The trainer will store this in its
        memory buffer for later use in policy updates.

        This method should be called after executing an action in the environment,
        once the reward and next state are known. The transition data will be
        used by the trainer's fit_model() to compute advantages and update the
        actor-critic network.

        Args:
            observation (np.ndarray): State observation at timestep t. Shape
                matches environment's observation space. This is s_t, the state
                before taking the action.
            action (int): Action taken at timestep t. Discrete action index
                in [0, num_actions). This is a_t, selected by the policy.
            reward (float): Immediate reward received at timestep t. Scalar
                value r_t returned by environment.step(). May be positive,
                negative, or zero.
            done (bool): Episode termination flag. True if the episode ended
                after this transition (terminal state reached), False otherwise.
                Used to zero out bootstrapped values in return computation.
            value (float): Critic's estimate of state value V(s_t). This is the
                value predicted by the trainer when the action was requested.
                Used for advantage calculation: A = R - V.

        Side Effects:
            * Sends JSON message over TCP connection.
            * Trainer stores transition in memory[port] buffer.

        Message Format:
            {
              "message_type": "step",
              "observation": [[...]],
              "action": 2,
              "reward": 1.0,
              "done": false,
              "value": 0.45
            }

        Notes:
            * The observation is s_t (pre-action), not s_{t+1} (post-action).
            * The value V(s_t) must match the value received with the action.
            * done=True signals the end of an episode; the trainer will not
              bootstrap from V(s_{t+1}) for this transition.
            * All transitions are stored, including terminal ones (done=True).
        """
        message = json.dumps(
            {
                "message_type": "step",
                "observation": observation.tolist(),
                "action": action,
                "reward": reward,
                "done": done,
                "value": value,
            }
        )
        tcp.send(self.connection, message)

    def tcp_send_terminated(self) -> None:
        """
        Send termination signal to trainer and prepare for disconnection.

        Notifies the trainer that this agent has completed all its episodes and
        is ready to disconnect. The trainer will remove this agent from its active
        connection pool and may trigger a final model checkpoint if all agents
        have terminated.

        This method should be called once at the end of the run() loop, after
        all epochs have been completed. It allows the trainer to perform cleanup
        and determine when to stop the overall training process.

        Side Effects:
            * Sends JSON message over TCP connection.
            * Trainer logs the termination and removes the connection.
            * Trainer may save the model if this is the last active agent.

        Message Format:
            {"message_type": "terminated"}

        Notes:
            * No response is expected from the trainer.
            * The socket connection may be closed shortly after sending.
            * The trainer's run() loop will detect the termination and update
              its active socket list.
            * If all agents terminate, the trainer saves the model and exits.
        """
        message = json.dumps(
            {
                "message_type": "terminated",
            }
        )
        tcp.send(self.connection, message)
