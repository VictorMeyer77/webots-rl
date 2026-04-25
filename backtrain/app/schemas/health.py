"""Health monitoring schemas for system components.

This module defines Pydantic schemas used for health check endpoints and system
monitoring. It provides structured data models for memory statistics and overall
system health information.

The schemas are used by health check endpoints to return consistent, validated
responses about the state of memory channels (action, environment, observation)
and the supervisor component.
"""

from app.schemas.supervisor import TrainingSchema
from pydantic import BaseModel


class MemoryStatsSchema(BaseModel):
    """Memory statistics for communication channels.

    Represents real-time health metrics for memory-based communication channels
    (action, environment, observation). Provides insights into current usage,
    capacity, and availability for monitoring and capacity planning.

    Attributes:
        size: Current number of messages stored in the memory channel.
        capacity: Maximum number of messages that can be stored before eviction.
        remaining: Available slots before memory reaches capacity (capacity - size).
        usage_percent: Percentage of capacity currently used, ranging from 0.0 to 100.0.
    """

    size: int
    capacity: int
    remaining: int
    usage_percent: float


class SystemInfoSchema(BaseModel):
    """System-wide health information.

    Aggregates health statistics from all system components including the
    supervisor and memory channels. Used for comprehensive system monitoring
    and health checks.

    Attributes:
        supervisor: List of active training sessions with their worker details.
        action_memory: Health statistics for the action communication channel.
        environment_memory: Health statistics for the environment communication channel.
        observation_memory: Health statistics for the observation communication channel.
    """

    supervisor: list[TrainingSchema]
    action_memory: MemoryStatsSchema
    environment_memory: MemoryStatsSchema
    observation_memory: MemoryStatsSchema
