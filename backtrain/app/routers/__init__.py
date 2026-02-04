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
from app.schemas import SystemInfoSchema
from fastapi import APIRouter, Depends

router = APIRouter()


@router.get(
    "/health",
    response_model=SystemInfoSchema,
    summary="System health check",
    description="Get comprehensive health status of all system components including training sessions and memory stores.",
    response_description="Health metrics for supervisor and all memory components",
    tags=["Health"],
    status_code=200,
    responses={
        200: {
            "description": "Health check successful",
            "content": {
                "application/json": {
                    "example": {
                        "supervisor": {"train_001": {0: 5, 1: 3}, "train_002": {0: 10}},
                        "action_memory": {
                            "size": 1000,
                            "capacity": 10000,
                            "remaining": 9000,
                            "usage_percent": 10.0,
                        },
                        "observation_memory": {
                            "size": 1500,
                            "capacity": 10000,
                            "remaining": 8500,
                            "usage_percent": 15.0,
                        },
                        "environment_memory": {
                            "size": 800,
                            "capacity": 10000,
                            "remaining": 9200,
                            "usage_percent": 8.0,
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
    """Get health status of all system components.

    Returns comprehensive health information about:
    - Supervisor: Current training sessions and worker states
    - Action Memory: Storage statistics and usage metrics
    - Observation Memory: Storage statistics and usage metrics
    - Environment Memory: Storage statistics and usage metrics

    Args:
        supervisor: Supervisor instance provided by dependency injection.
        action_memory: Action memory store provided by dependency injection.
        observation_memory: Observation memory store provided by dependency injection.
        environment_memory: Environment memory store provided by dependency injection.

    Returns:
        Dictionary containing health metrics for all components:
        - 'supervisor': Training session data with worker assignments and episode IDs
        - 'action_memory': Statistics about action memory usage
        - 'observation_memory': Statistics about observation memory usage
        - 'environment_memory': Statistics about environment memory usage

    HTTP Status Codes:
        200: Health check successful, all components accessible
    """
    return SystemInfoSchema(
        supervisor=supervisor.training,
        action_memory=action_memory.stats(),
        observation_memory=observation_memory.stats(),
        environment_memory=environment_memory.stats(),
    )
