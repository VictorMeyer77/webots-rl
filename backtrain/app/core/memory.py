"""Memory Module for Distributed RL Systems.

This module provides a lightweight, capacity-limited memory store that facilitates
asynchronous communication between different components in a distributed
reinforcement learning architecture. It acts as a message broker or shared cache
for exchanging RL data between decoupled services.

Architecture Overview:
    This memory acts as a communication layer between:
    - Environment: Generate state and rewards
    - Agent: Produce actions based on observations
    - Trainer: Consume experiences for model updates

    Each component can publish data to the memory, and other components can subscribe
    by querying specific keys.


Message Routing:
    The hierarchical key structure enables efficient message routing:
    - train_id (str): Training session/experiment identifier
    - worker_id (int): Parallel running instance (for multi-env setups)
    - episode_id (int): Episode number within session
    - step (int): Time step within episode

    This allows components to:
    - Subscribe to specific training sessions
    - Handle parallel workers independently
    - Process episodes in sequence or parallel
    - Access individual time steps for debugging

Memory Management:
    - **Transient storage**: Not persistent across restarts
    - **LRU eviction**: Oldest messages removed when capacity exceeded
    - **No durability**: Messages lost if not consumed before eviction
    - **Single-instance**: Not distributed; for distributed setups, use Redis/RabbitMQ

Type Safety:
    Three message types are supported:
    - EnvironmentSchema: Environment responses (state, reward, done)
    - ObservationSchema: Observations sent by agents
    - ActionSchema: Actions from trainers to agents

    Each memory instance typically handles one message type for clarity,
    but can mix types if needed for flexibility.
"""

from collections import OrderedDict

from app.schemas import ActionSchema, EnvironmentSchema, ObservationSchema
from app.schemas.health import MemoryStatsSchema

Key = tuple[str, int, int, int]


class Memory:
    """Lightweight message store for communication in RL systems.

    This class provides transient, capacity-limited storage for messages exchanged
    between decoupled RL system components. It enables asynchronous
    communication patterns like publish-subscribe and request-response without
    requiring persistent message brokers.

    The memory uses LRU (Least Recently Used) eviction to maintain bounded memory
    usage. When capacity is exceeded, the oldest unread messages are discarded.

    Attributes:
        capacity (int): Maximum number of messages to store before eviction.
        memory (OrderedDict): Internal ordered storage mapping keys to message payloads.
            Keys are hierarchical tuples for message routing.
            Values are typed message schemas (Environment/Observation/Action).
    """

    def __init__(self, capacity: int):
        """Initialize a message store.

        Args:
            capacity (int): Maximum number of messages to store. Must be positive.
                Determines memory footprint. When exceeded, oldest messages are evicted.
        """
        self.capacity = capacity
        self.memory: OrderedDict[
            Key, EnvironmentSchema | ObservationSchema | ActionSchema
        ] = OrderedDict()

    def add(
        self, key: Key, value: EnvironmentSchema | ObservationSchema | ActionSchema
    ) -> None:
        """Publish a message to the communication channel.

        This method publishes a message that other bricks can consume. If the key
        already exists, the message is updated (overwrites previous value). When
        capacity is exceeded, the oldest unpublished message is evicted.

        Args:
            key (Key): Hierarchical routing key (train_id, worker_id, episode_id, step).
            value: Message payload (EnvironmentSchema, ObservationSchema, or ActionSchema).
        """
        self.memory[key] = value
        if len(self.memory) > self.capacity:
            self.memory.popitem(last=False)

    def get(
        self, key: Key
    ) -> EnvironmentSchema | ObservationSchema | ActionSchema | None:
        """Consume a message from the communication channel.

        Retrieves a message published. This is a non-destructive
        read; the message remains available for other consumers.

        Args:
            key (Key): Hierarchical routing key to retrieve.

        Returns:
            EnvironmentSchema | ObservationSchema | ActionSchema | None:
                The message payload if found, None if not available.
        """

        return self.memory.get(key, None)

    def __contains__(self, key: Key) -> bool:
        """Check if a message exists in the communication channel.

        Tests message availability without retrieving it. Useful for polling
        or conditional processing.

        Args:
            key (Key): Routing key to check.

        Returns:
            bool: True if message available, False otherwise.
        """
        return key in self.memory

    def __len__(self) -> int:
        """Get the current number of messages in the communication channel.

        Returns:
            int: Number of messages currently stored.
        """
        return len(self.memory)

    def clear(self) -> None:
        """Clear all messages from the communication channel.

        Removes all pending messages, effectively resetting the communication
        state between bricks. Useful for starting new training sessions or
        recovering from error states.
        """
        self.memory.clear()

    def stats(self) -> MemoryStatsSchema:
        """Get memory usage statistics for monitoring and capacity planning.

        Provides real-time metrics about the communication channel's current state,
        useful for monitoring system health, detecting backpressure, and optimizing
        capacity allocation across bricks.

        Returns:
            MemoryStatsSchema: Memory statistics containing:
                - size (int): Current number of stored messages
                - capacity (int): Maximum number of messages that can be stored
                - remaining (int): Available slots before eviction occurs (capacity - size)
                - usage_percent (float): Percentage of capacity currently used (0.0 to 100.0)
        """
        return MemoryStatsSchema(
            size=len(self.memory),
            capacity=self.capacity,
            remaining=self.capacity - len(self.memory),
            usage_percent=(
                round((len(self.memory) / self.capacity * 100), 2)
                if self.capacity > 0
                else 0.0
            ),
        )
