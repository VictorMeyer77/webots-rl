from enum import StrEnum


class Endpoint(StrEnum):
    """
    API endpoint identifiers for the RL training system.

    This enum defines the available REST API endpoints for communication
    between workers, environments, and the central training server.

    Attributes:
        ACTION: Endpoint for storing/retrieving agent actions
        OBSERVATION: Endpoint for storing/retrieving environment observations
        ENVIRONMENT: Endpoint for storing/retrieving environment state (reward, done)
        SUPERVISOR: Endpoint for training coordination (workers, episodes, etc.)
    """

    ACTION = "action"
    OBSERVATION = "observation"
    ENVIRONMENT = "environment"
    SUPERVISOR = "supervisor"
