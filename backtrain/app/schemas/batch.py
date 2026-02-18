"""Batch operation schemas for bulk requests and responses.

This module provides Pydantic schemas for batch operations across all
communication channels (action, environment, observation), enabling efficient
bulk retrieval and publishing of multiple items in single API calls.
"""

from app.schemas import (
    ActionSchema,
    EnvironmentSchema,
    ObservationSchema,
    StepKeySchema,
)
from pydantic import BaseModel


class BatchGetRequestSchema(BaseModel):
    """
    Represents a batch request to retrieve multiple items.

    Attributes:
        keys: List of keys to retrieve (actions, environments, or observations).
    """

    keys: list[StepKeySchema]


class BatchItemSchema(BaseModel):
    """
    Represents a single item in a batch request or response.

    Attributes:
        key: The key identifying the item.
        value: The data (Action, Environment, or Observation) if found, None otherwise.
    """

    key: StepKeySchema
    value: ActionSchema | EnvironmentSchema | ObservationSchema | None


class BatchGetResponseSchema(BaseModel):
    """
    Represents the response for a batch retrieval request.

    Attributes:
        results: List of results corresponding to the requested keys.
        total: Total number of keys requested.
        found: Number of items successfully found.
        missing: Number of items not found.
    """

    results: list[BatchItemSchema]
    total: int
    found: int
    missing: int


class BatchPostRequestSchema(BaseModel):
    """
    Represents a batch request to publish multiple items.

    Attributes:
        items: List of items (key-value pairs) to publish.
    """

    items: list[BatchItemSchema]


class BatchPostResponseSchema(BaseModel):
    """
    Represents the response for a batch publish request.

    Attributes:
        status: Status message (defaults to "success").
        total: Total number of items published.
    """

    status: str = "success"
    total: int
