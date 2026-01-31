"""
FastAPI dependency injection functions for memory management.

This module provides dependency functions that retrieve memory instances from
the application state. These dependencies are used to inject memory storage
into API route handlers for managing environment states, observations, and actions.
"""

from fastapi import Request


def get_environment_memory(request: Request):
    """
    Retrieve the environment memory.

    This dependency function provides access to the memory storage that holds
    environment states.

    Args:
        request: The FastAPI request object containing the application state.

    Returns:
        The environment memory instance stored in the application state.
    """
    return request.app.state.environment_memory


def get_observation_memory(request: Request):
    """
    Retrieve the observation memory.

    This dependency function provides access to the memory storage that holds
    observations.

    Args:
        request: The FastAPI request object containing the application state.

    Returns:
        The observation memory instance stored in the application state.
    """
    return request.app.state.observation_memory


def get_action_memory(request: Request):
    """
    Retrieve the action memory.

    This dependency function provides access to the memory storage that holds
    actions.

    Args:
        request: The FastAPI request object containing the application state.

    Returns:
        The action memory instance stored in the application state.
    """
    return request.app.state.action_memory
