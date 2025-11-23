"""
Controller-side e-puck tabular Q-table integration.

This module implements the robot-facing counterpart of a tabular RL setup
using a shared Q-table: the supervisor (trainer) drives learning, while
this controller:
  1. Waits for a synchronization message from the supervisor (``{'sync': 1}``)
     and replies with ``{'ack': 1}``.
  2. Sends discretized distance sensor observations (``{'observation': {...}}``).
  3. Receives an action (``{'action': int}``) selected by the supervisor policy.
  4. Executes the action and reports step completion (``{'step': step_index}``).

Observation discretization (per distance sensor value):
  value > 80 -> bin 0 (far)
  value > 70 -> bin 1 (medium)
  else        -> bin 2 (near)
"""

import numpy as np
from brain.controller.epuck import EpuckTurner
from brain.utils.logger import logger


class EpuckTurnerQTable(EpuckTurner):
    """E-puck controller specialization for tabular Q-table control.

    This controller is algorithm-agnostic: it can be paired with SARSA,
    Q-learning, Monte Carlo, or any tabular method that maintains a
    shared Q-table on the supervisor side.

    Responsibilities:
      * Discretize raw distance sensor readings for compatibility with a
        tabular Q-table.
      * Provide a greedy ``policy()`` used during evaluation (supervisor
        handles exploration during training).
      * Run a message-driven training loop (``train()``) coordinating
        with the supervisor.
    """

    def observe(self) -> dict:
        """Collect raw distance sensor readings and discretize them.

        Binning:
            > 80 -> 0 (far)
            > 70 -> 1 (medium)
            else -> 2 (near)

        Returns:
            dict: ``{'distance_sensors': list[int]}`` binned distances.
        """
        observations = super().observe()
        bins = {"distance_sensors": []}
        for distance in observations["distance_sensors"]:
            if distance > 80:
                bins["distance_sensors"].append(0)
            elif distance > 70:
                bins["distance_sensors"].append(1)
            else:
                bins["distance_sensors"].append(2)
        return bins

    def policy(self, observation: dict) -> int:
        """Select greedy action for the current binned observation.

        Args:
            observation (dict): Output from :meth:`observe`.

        Returns:
            int: Action index with maximal Q-value for the given state.
        """
        index = self.model.observation_to_index(np.array(observation["distance_sensors"]))
        action_values = self.model.q_table[index]
        action = np.argmax(action_values)
        return int(action)

    def train(self) -> None:
        """Execute the controller-side training communication loop.

        Sequence per iteration:
          1. Synchronize: wait for ``{'sync': 1}`` then reply ``{'ack': 1}``
             once.
          2. Observation: send discretized sensor data.
          3. Action reception: wait for supervisor-selected action.
          4. Actuation: apply action via :meth:`act` and notify completion
             with ``{'step': step_index}``.
          5. Repeat until simulation ends (``robot.step()`` returns ``-1``).

        Notes:
          * Exploration (epsilon-greedy) is handled supervisor-side; the
            controller always executes the received actions.
          * The controller does not update the Q-table; it only supplies
            observations and executes actions.
        """
        step_index = 0
        sync = False
        step_observation = None
        step_action = None

        while self.robot.step(self.timestep) != -1:

            self.queue.clear_buffer()

            # (1) Initial synchronization handshake on the very first step.
            if not sync:
                if not self.queue.search_message("sync"):
                    continue
                else:
                    self.queue.send({"ack": 1})
                    sync = True
                    logger().debug("Synchronization with supervisor successful.")

            # (2) Send observation.
            if step_observation is None:
                step_observation = self.observe()
                self.queue.send({"observation": step_observation})

            # (3) Await action.
            if step_action is None:
                action_messages = self.queue.search_message("action")
                if not action_messages:
                    continue
                else:
                    step_action = action_messages[0]["action"]
                    logger().debug(f"Received action {step_action}")

            # (4) Execute action and notify step completion.
            self.act(step_action)
            self.queue.send({"step": step_index})

            # (5) Prepare for next iteration.
            step_index += 1
            step_observation = None
            step_action = None
            logger().debug(f"Controller completed step {step_index}")
