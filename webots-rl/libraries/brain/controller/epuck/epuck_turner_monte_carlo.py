"""
Monte Carlo policy-based e-puck turning controller.

Loads a precomputed NumPy Q-table mapping discretized distance sensor observations
to action values and selects greedy actions each control step.

Workflow:
1. observe(): Gather raw distance sensor readings via superclass; bin each into {0,1,2}.
2. observation_to_index(): Encode binned list into a linear Q-table index (mixed radix).
3. policy(): Perform greedy argmax selection over Q-values for current state.
4. train(): Supervisor-synchronized loop exchanging 'sync', 'observation', 'action', and 'step' messages.

Notes:
- Q-table path is hard-coded; parameterize for portability.
- observation_cardinality defines uniform bin count per sensor.
"""

import numpy as np
from brain.controller.epuck import EpuckTurner
from brain.utils.logger import logger


class EpuckTurnerMonteCarlo(EpuckTurner):
    """
    E-puck turning controller using a precomputed Monte Carlo Q-table.
    """

    def observe(self) -> dict:
        """
        Collect raw distance sensor readings and discretize them.

        Binning:
            > 80 -> 0 (far)
            > 70 -> 1 (medium)
            else -> 2 (near)

        Returns:
            dict: {'distance_sensors': list[int]} binned distances.
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
        """
        Select greedy action for the current binned observation.

        Args:
            observation (dict): Output from observe().

        Returns:
            int: Action index with maximal Q-value.
        """
        index = self.model.observation_to_index(np.array(observation["distance_sensors"]))
        action_values = self.model.q_table[index]
        action = np.argmax(action_values)
        return int(action)

    def train(self) -> None:
        """
        Supervisor-synchronized main loop.

        Protocol:
            1. Await 'sync' message; respond with {'ack': 1} once.
            2. Each timestep:
                a. Send {'observation': binned_observation}.
                b. Await single 'action' message; ignore until available.
                c. Execute action via act().
                d. Send {'step': step_index}.
            3. Repeat until simulation ends.

        Side Effects:
            - Sends and receives messages over self.queue.
            - Commands robot motors via act().

        Notes:
            - Buffer is cleared each iteration to reduce stale message buildup.
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
