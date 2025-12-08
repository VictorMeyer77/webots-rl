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
