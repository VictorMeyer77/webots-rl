import logging
import time
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from controller import Robot

from corl.api.wrapper import Wrapper
from corl.schemas.learning import Action, Observation
from corl.utils.config import Config

logger = logging.getLogger(__name__)


class Agent(ABC):
    """
    Abstract base class for a Webots reinforcement learning agent.

    Subclasses must implement :meth:`observe`, :meth:`policy`, and :meth:`act`,
    which together form the perception-decision-action cycle driven by
    :meth:`step`.

    Two execution modes are supported:

    - **Run mode** (``config["train"]`` is falsy): drives the robot with
      :meth:`run` until the Webots simulation stops.
    - **Training mode** (``config["train"]`` is truthy): drives a single
      training episode via :meth:`train`, exchanging observations, actions,
      and acknowledgments with the remote trainer through the
      :class:`~corl.api.wrapper.Wrapper` API.

    Attributes:
        robot: Webots ``Robot`` node being controlled.
        timestep: Simulation timestep in milliseconds.
        model: Optional loaded model used by :meth:`policy` (run mode).
        train_id: Training session identifier (training mode only).
        worker_id: Worker identifier within the training session (training mode only).
        request_interval_steps: How often (in simulation timesteps) to poll
            the API for a new action (training mode only).
        api: :class:`~corl.api.wrapper.Wrapper` instance for API communication
            (training mode only).
    """

    robot: Robot
    timestep: int
    model: tf.keras.Model | np.ndarray | None = None
    # Training attributes
    train_id: str | None = None
    worker_id: int | None = None
    request_interval_steps: int | None = None
    api: Wrapper | None = None

    def __init__(
        self,
        robot: Robot,
        config: Config,
        timestep: int = 64,
    ):
        """
        Initialize the agent.

        In run mode (``config["train"]`` is falsy) only the robot and timestep
        attributes are set; all training attributes remain ``None``.

        In training mode (``config["train"]`` is truthy) the training
        attributes are populated and a :class:`~corl.api.wrapper.Wrapper`
        instance is created.

        Args:
            robot: Webots ``Robot`` node to control.
            config: Application configuration. Must contain ``"train"`` and,
                when ``"train"`` is truthy, ``"train_id"``, ``"worker_id"``,
                and ``"get_request_interval_steps"``.
            timestep: Simulation timestep in milliseconds (default: ``64``).
        """
        self.robot = robot
        self.timestep = timestep

        if config["train"]:
            self.train_id = config["train_id"]
            self.worker_id = config["worker_id"]
            self.request_interval_steps = config["get_request_interval_steps"]
            self.api = Wrapper(config)
            logger.debug(
                f"Agent initialized for training with train_id={self.train_id} and worker_id={self.worker_id}"
            )

    @abstractmethod
    def observe(self) -> dict:
        """
        Read robot sensors and build an observation payload.

        Subclasses must override this method to sample all relevant sensors
        and return their readings as a plain dictionary suitable for passing
        to :meth:`policy`.

        Returns:
            dict: Structured sensor data for policy consumption.

        Raises:
            NotImplementedError: Always, if the subclass does not override
                this method.
        """
        raise NotImplementedError("Method observe() not implemented.")

    @abstractmethod
    def policy(self, observation: dict) -> int:
        """
        Decide an action based on the current observation.

        Subclasses must override this method to implement the agent's
        decision logic, whether rule-based or model-based.

        Args:
            observation: Sensor-derived observation returned by :meth:`observe`.

        Returns:
            int: Discrete action identifier to be passed to :meth:`act`.

        Raises:
            NotImplementedError: Always, if the subclass does not override
                this method.
        """
        raise NotImplementedError("Method policy() not implemented.")

    @abstractmethod
    def act(self, action: int) -> None:
        """
        Execute the chosen action on the robot.

        Subclasses must override this method to translate the discrete action
        identifier into concrete motor or actuator commands.

        Args:
            action: Discrete action identifier returned by :meth:`policy`.

        Raises:
            NotImplementedError: Always, if the subclass does not override
                this method.
        """
        raise NotImplementedError("Method act() not implemented.")

    def step(self) -> None:
        """
        Perform one perception-decision-action cycle.

        Calls :meth:`observe` to read sensors, passes the result to
        :meth:`policy` to select an action, then calls :meth:`act` to
        execute it.
        """
        observation = self.observe()
        action = self.policy(observation)
        self.act(action)

    def load_model(self, model: tf.keras.Model | np.ndarray) -> None:
        """
        Load a trained model into the agent for use by :meth:`policy`.

        Args:
            model: A Keras model or NumPy array representing the policy
                weights or lookup table.
        """
        self.model = model
        logger.info("Model loaded into agent.")

    def run(self) -> None:
        """
        Continuous control loop until simulation termination.

        Drives the Webots simulation step-by-step. On each iteration,
        :meth:`step` is called to complete one perception-decision-action
        cycle. The loop exits when ``robot.step()`` returns ``-1``,
        indicating that the Webots simulation has stopped.
        """
        step = 0
        while self.robot.step(self.timestep) != -1:
            self.step()
            step += 1
            logger.debug(f"Agent completed step {step}")

    def train(self) -> None:
        """
        Execute a single training episode.

        Runs the Webots simulation loop and synchronises with the remote
        trainer at each logical RL step:

        1. **Observation dispatch** — calls :meth:`observe` and sends the
           result to the API via ``api.send_observation()``, retrying each
           timestep until acknowledged.
        2. **Action retrieval and execution** — polls ``api.get_action()``
           every ``request_interval_steps`` timesteps until an action is
           returned, then calls :meth:`act` to execute it.
        3. **Action acknowledgment** — sends a confirmed
           :class:`~corl.schemas.learning.Action` back to the API via
           ``api.send_action()``, retrying until successful.

        After the simulation loop exits (``robot.step()`` returns ``-1``),
        a summary is logged with the episode identifier, step count, and
        wall-clock duration.

        Requires the agent to have been initialized in training mode
        (i.e. ``config["train"]`` was truthy at construction time).
        """
        step: int = 0
        observation: dict | None = None
        observation_ack: bool = False
        action: Action | None = None
        start_counter: float = time.perf_counter()
        episode_id: int = self.api.get_episode_id(
            self.train_id, worker_id=self.worker_id
        )
        timestep_count: int = 0
        step_timestep_count: int = 0

        while self.robot.step(self.timestep) != -1:
            timestep_count += 1
            step_timestep_count += 1

            # (1) Send observation.
            if observation is None:
                observation = self.observe()

            if not observation_ack:
                if self.api.send_observation(
                    self.train_id,
                    self.worker_id,
                    episode_id,
                    step,
                    Observation(data=observation),
                ):
                    observation_ack = True
                else:
                    continue

            # (2) Await and execute action.
            if action is None:
                if (step_timestep_count - 1) % self.request_interval_steps == 0:
                    action = self.api.get_action(
                        self.train_id, self.worker_id, episode_id, step
                    )
                    if action is not None:
                        self.act(action.action)
                    else:
                        continue
                else:
                    continue

            # (3) Send action acknowledgment.
            if not self.api.send_action(
                self.train_id,
                self.worker_id,
                episode_id,
                step,
                Action(action=action.action, executed=True),
            ):
                continue

            # (4) Prepare for next iteration.
            step += 1
            step_timestep_count = 0
            observation = None
            observation_ack = False
            action = None
            logger.debug(f"Completed training step {step} for episode {episode_id}")

        logger.info(
            f"{self.worker_id} completed episode {episode_id} with {step} steps. Duration: {(time.perf_counter() - start_counter):.6f} seconds and {timestep_count} timesteps."
        )
