from brain.environment import Environment


class EnvironmentGenetic(Environment):
    """
    Base environment class for genetic algorithm-based simulations.

    Inherits from:
        Environment

    Attributes:
        actions (list[float]): List of actions to be executed by the agent.
    """

    actions: list[float] = []

    def set_actions(self, actions: list[float]):
        """
        Set the actions for the agent.

        Args:
            actions (list[float]): The actions to be executed by the agent.
        """
        self.actions = actions
