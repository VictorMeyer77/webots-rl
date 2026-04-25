"""
FastAPI application entry point and initialization.

This module sets up the FastAPI application instance, configures logging,
initializes memory storage for reinforcement learning data, and registers
API routers for managing environment states, observations, actions, and
experiences. The application uses a lifespan context manager to initialize
memory instances with configurable capacity settings.

Logging is configured at application startup to provide consistent formatting
and level management across all modules. The logging configuration supports
both console and file handlers with independent log levels.
"""

from contextlib import asynccontextmanager

import uvicorn
from app.core.config import settings
from app.core.logger import get_logger, setup_logging
from app.core.memory import Memory
from app.core.supervisor import Supervisor
from app.routers import router as root_router
from app.routers.action import router as action_router
from app.routers.environment import router as environment_router
from app.routers.observation import router as observation_router
from app.routers.supervisor import router as supervisor_router
from fastapi import FastAPI

# Configure application-wide logging
setup_logging()
logger = get_logger(__name__)

API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the application lifespan and initialize memory instances.

    This async context manager handles the startup and shutdown of the FastAPI
    application. During startup, it creates three memory instances with the
    configured capacity and attaches them to the application state. These
    memory instances are used to store environment data, observations, and
    actions throughout the application's lifetime.

    The memory capacity can be configured through environment variables or
    a .env file by setting the MEMORY_CAPACITY variable.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the application runtime after initialization.
    """

    app.state.environment_memory = Memory(capacity=settings.memory_capacity)
    app.state.observation_memory = Memory(capacity=settings.memory_capacity)
    app.state.action_memory = Memory(capacity=settings.memory_capacity)
    app.state.supervisor = Supervisor()
    yield


# Initialize the FastAPI application with lifespan management
app = FastAPI(lifespan=lifespan)

# Register API routers for handling different aspects of the RL system
app.include_router(root_router, prefix=API_PREFIX)
app.include_router(environment_router, prefix=API_PREFIX)
app.include_router(observation_router, prefix=API_PREFIX)
app.include_router(action_router, prefix=API_PREFIX)
app.include_router(supervisor_router, prefix=API_PREFIX)


if __name__ == "__main__":
    # Get your custom logger
    logger = get_logger("uvicorn")

    # Configure uvicorn to use your logger
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn"]["handlers"] = []
    log_config["loggers"]["uvicorn.access"]["handlers"] = []
    log_config["loggers"]["uvicorn.error"]["handlers"] = []

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,  # Disable default logging
        log_level=0,
    )
