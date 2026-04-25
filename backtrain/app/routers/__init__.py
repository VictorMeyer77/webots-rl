"""Router initialization module.

This module initializes and aggregates all API routers for the application.
It provides root-level endpoints including health checks for monitoring system status.
"""

from app.core.memory import Memory
from app.core.supervisor import Supervisor
from app.dependencies import (
    get_action_memory,
    get_environment_memory,
    get_observation_memory,
    get_supervisor,
)
from app.schemas.health import SystemInfoSchema
from fastapi import APIRouter, Depends

router = APIRouter()


@router.get(
    "/health",
    response_model=SystemInfoSchema,
    summary="System health and status check",
    description="Retrieve comprehensive system health information including all training sessions, "
    "workers, and memory usage statistics across all communication channels. "
    "This endpoint provides a complete overview of the system's current state, "
    "useful for monitoring, debugging, and capacity planning.",
    responses={
        200: {
            "description": "System health information retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "supervisor": [
                            {
                                "id": "train_001",
                                "workers": [
                                    {"id": 0, "episode_id": 42, "status": True},
                                    {"id": 1, "episode_id": 38, "status": False},
                                ],
                            }
                        ],
                        "action_memory": {
                            "size": 150,
                            "capacity": 1000,
                            "remaining": 850,
                            "usage_percent": 15.0,
                        },
                        "observation_memory": {
                            "size": 200,
                            "capacity": 1000,
                            "remaining": 800,
                            "usage_percent": 20.0,
                        },
                        "environment_memory": {
                            "size": 100,
                            "capacity": 1000,
                            "remaining": 900,
                            "usage_percent": 10.0,
                        },
                    }
                }
            },
        }
    },
)
async def health(
    supervisor: Supervisor = Depends(get_supervisor),
    action_memory: Memory = Depends(get_action_memory),
    observation_memory: Memory = Depends(get_observation_memory),
    environment_memory: Memory = Depends(get_environment_memory),
) -> SystemInfoSchema:
    """Retrieve comprehensive system health and status information.

    Provides a complete snapshot of the system's current state, including:
    - All active training sessions with their registered workers
    - Worker status and episode progress for each training session
    - Memory usage statistics for action, observation, and environment channels
    - Capacity metrics and remaining space for each memory component

    This endpoint is designed for:
    - Health monitoring and alerting systems
    - Capacity planning and resource optimization
    - Debugging training session issues
    - Real-time system status dashboards

    Args:
        supervisor: Supervisor instance managing all training sessions.
        action_memory: Memory instance for agent action messages.
        observation_memory: Memory instance for environment observation messages.
        environment_memory: Memory instance for environment state messages.

    Returns:
        Complete system information including supervisor state and memory statistics
        for all communication channels.
    """
    return SystemInfoSchema(
        supervisor=supervisor.trainings,
        action_memory=action_memory.stats(),
        observation_memory=observation_memory.stats(),
        environment_memory=environment_memory.stats(),
    )
