"""Unit tests for the Memory module."""

import pytest
from app.core.memory import Key, Memory
from app.schemas import ActionSchema, EnvironmentSchema, ObservationSchema
from app.schemas.health import MemoryStatsSchema


@pytest.fixture
def sample_key() -> Key:
    """Provide a sample routing key for tests."""
    return ("train_001", 0, 1, 10)


@pytest.fixture
def sample_environment_message() -> EnvironmentSchema:
    """Provide a sample environment message."""
    return EnvironmentSchema(
        done=False, reward=1.0, data={"position": [0.1, 0.2], "velocity": 0.5}
    )


@pytest.fixture
def sample_observation_message() -> ObservationSchema:
    """Provide a sample observation message."""
    return ObservationSchema(
        data={"sensor_readings": [0.5, 0.6, 0.7], "timestamp": 100}
    )


@pytest.fixture
def sample_action_message() -> ActionSchema:
    """Provide a sample action message."""
    return ActionSchema(action=2)


def test_memory_initialization():
    """Test that Memory is initialized with correct capacity."""
    memory = Memory(capacity=100)

    assert memory.capacity == 100
    assert len(memory) == 0
    assert isinstance(memory.memory, dict)


def test_memory_add_message(sample_key, sample_environment_message):
    """Test that messages can be added to memory."""
    memory = Memory(capacity=10)

    memory.add(sample_key, sample_environment_message)

    assert len(memory) == 1
    assert sample_key in memory


def test_memory_add_multiple_messages(sample_environment_message):
    """Test that multiple messages can be added with different keys."""
    memory = Memory(capacity=10)
    key1 = ("train_001", 0, 1, 10)
    key2 = ("train_001", 0, 1, 11)
    key3 = ("train_001", 1, 1, 10)

    memory.add(key1, sample_environment_message)
    memory.add(key2, sample_environment_message)
    memory.add(key3, sample_environment_message)

    assert len(memory) == 3
    assert key1 in memory
    assert key2 in memory
    assert key3 in memory


def test_memory_add_overwrites_existing_key(
    sample_key, sample_environment_message, sample_observation_message
):
    """Test that adding with existing key overwrites the previous value."""
    memory = Memory(capacity=10)

    memory.add(sample_key, sample_environment_message)
    memory.add(sample_key, sample_observation_message)

    assert len(memory) == 1
    retrieved = memory.get(sample_key)
    assert isinstance(retrieved, ObservationSchema)


def test_memory_get_existing_message(sample_key, sample_environment_message):
    """Test that existing messages can be retrieved."""
    memory = Memory(capacity=10)
    memory.add(sample_key, sample_environment_message)

    retrieved = memory.get(sample_key)

    assert retrieved == sample_environment_message
    assert retrieved.reward == 1.0
    assert retrieved.done is False


def test_memory_get_nonexistent_message():
    """Test that retrieving nonexistent message returns None."""
    memory = Memory(capacity=10)
    key = ("train_001", 0, 1, 10)

    retrieved = memory.get(key)

    assert retrieved is None


def test_memory_get_is_nondestructive(sample_key, sample_environment_message):
    """Test that get does not remove the message from memory."""
    memory = Memory(capacity=10)
    memory.add(sample_key, sample_environment_message)

    memory.get(sample_key)
    memory.get(sample_key)

    assert len(memory) == 1
    assert sample_key in memory


def test_memory_contains_existing_key(sample_key, sample_environment_message):
    """Test that __contains__ returns True for existing keys."""
    memory = Memory(capacity=10)
    memory.add(sample_key, sample_environment_message)

    assert sample_key in memory


def test_memory_contains_nonexistent_key():
    """Test that __contains__ returns False for nonexistent keys."""
    memory = Memory(capacity=10)
    key = ("train_001", 0, 1, 10)

    assert key not in memory


def test_memory_len_empty():
    """Test that __len__ returns 0 for empty memory."""
    memory = Memory(capacity=10)

    assert len(memory) == 0


def test_memory_len_with_messages(sample_environment_message):
    """Test that __len__ returns correct count of messages."""
    memory = Memory(capacity=10)

    for i in range(5):
        memory.add(("train_001", 0, 1, i), sample_environment_message)

    assert len(memory) == 5


def test_memory_clear(sample_environment_message):
    """Test that clear removes all messages."""
    memory = Memory(capacity=10)

    for i in range(5):
        memory.add(("train_001", 0, 1, i), sample_environment_message)

    memory.clear()

    assert len(memory) == 0


def test_memory_lru_eviction(sample_environment_message):
    """Test that oldest messages are evicted when capacity is exceeded."""
    memory = Memory(capacity=3)
    key1 = ("train_001", 0, 1, 1)
    key2 = ("train_001", 0, 1, 2)
    key3 = ("train_001", 0, 1, 3)
    key4 = ("train_001", 0, 1, 4)

    memory.add(key1, sample_environment_message)
    memory.add(key2, sample_environment_message)
    memory.add(key3, sample_environment_message)
    memory.add(key4, sample_environment_message)

    assert len(memory) == 3
    assert key1 not in memory
    assert key2 in memory
    assert key3 in memory
    assert key4 in memory


def test_memory_lru_eviction_multiple(sample_environment_message):
    """Test that multiple oldest messages are evicted when adding several messages."""
    memory = Memory(capacity=2)

    for i in range(5):
        memory.add(("train_001", 0, 1, i), sample_environment_message)

    assert len(memory) == 2
    assert ("train_001", 0, 1, 0) not in memory
    assert ("train_001", 0, 1, 1) not in memory
    assert ("train_001", 0, 1, 2) not in memory
    assert ("train_001", 0, 1, 3) in memory
    assert ("train_001", 0, 1, 4) in memory


def test_memory_stats_empty():
    """Test that stats returns correct values for empty memory."""
    memory = Memory(capacity=100)

    stats = memory.stats()

    assert isinstance(stats, MemoryStatsSchema)
    assert stats.size == 0
    assert stats.capacity == 100
    assert stats.remaining == 100
    assert stats.usage_percent == 0.0


def test_memory_stats_partially_filled(sample_environment_message):
    """Test that stats returns correct values for partially filled memory."""
    memory = Memory(capacity=10)

    for i in range(3):
        memory.add(("train_001", 0, 1, i), sample_environment_message)

    stats = memory.stats()

    assert isinstance(stats, MemoryStatsSchema)
    assert stats.size == 3
    assert stats.capacity == 10
    assert stats.remaining == 7
    assert stats.usage_percent == 30.0


def test_memory_stats_full(sample_environment_message):
    """Test that stats returns correct values for full memory."""
    memory = Memory(capacity=5)

    for i in range(5):
        memory.add(("train_001", 0, 1, i), sample_environment_message)

    stats = memory.stats()

    assert isinstance(stats, MemoryStatsSchema)
    assert stats.size == 5
    assert stats.capacity == 5
    assert stats.remaining == 0
    assert stats.usage_percent == 100.0


def test_memory_stats_zero_capacity():
    """Test that stats handles zero capacity gracefully."""
    memory = Memory(capacity=0)

    stats = memory.stats()

    assert isinstance(stats, MemoryStatsSchema)
    assert stats.size == 0
    assert stats.capacity == 0
    assert stats.remaining == 0
    assert stats.usage_percent == 0.0


def test_memory_stats_usage_percent_rounding(sample_environment_message):
    """Test that usage_percent is rounded to 2 decimal places."""
    memory = Memory(capacity=3)
    memory.add(("train_001", 0, 1, 1), sample_environment_message)

    stats = memory.stats()

    assert isinstance(stats, MemoryStatsSchema)
    assert stats.usage_percent == 33.33


def test_memory_mixed_message_types(
    sample_environment_message, sample_observation_message, sample_action_message
):
    """Test that memory can store different message types."""
    memory = Memory(capacity=10)
    key1 = ("train_001", 0, 1, 1)
    key2 = ("train_001", 0, 1, 2)
    key3 = ("train_001", 0, 1, 3)

    memory.add(key1, sample_environment_message)
    memory.add(key2, sample_observation_message)
    memory.add(key3, sample_action_message)

    assert isinstance(memory.get(key1), EnvironmentSchema)
    assert isinstance(memory.get(key2), ObservationSchema)
    assert isinstance(memory.get(key3), ActionSchema)


def test_memory_environment_message_data_structure(sample_key):
    """Test that environment messages preserve complex data structures."""
    memory = Memory(capacity=10)
    env_msg = EnvironmentSchema(
        done=True,
        reward=5.5,
        data={"nested": {"values": [1, 2, 3]}, "metadata": "test"},
    )

    memory.add(sample_key, env_msg)
    retrieved = memory.get(sample_key)

    assert isinstance(retrieved, EnvironmentSchema)
    assert retrieved.done is True
    assert retrieved.reward == 5.5
    assert retrieved.data["nested"]["values"] == [1, 2, 3]


def test_memory_observation_message_data_structure(sample_key):
    """Test that observation messages preserve data structures."""
    memory = Memory(capacity=10)
    obs_msg = ObservationSchema(
        data={"sensors": [0.1, 0.2, 0.3], "camera": {"width": 640, "height": 480}}
    )

    memory.add(sample_key, obs_msg)
    retrieved = memory.get(sample_key)

    assert isinstance(retrieved, ObservationSchema)
    assert retrieved.data["sensors"] == [0.1, 0.2, 0.3]
    assert retrieved.data["camera"]["width"] == 640


def test_memory_action_message_values(sample_key):
    """Test that action messages store integer action identifiers correctly."""
    memory = Memory(capacity=10)

    for action_value in [0, 1, 5, 100]:
        action_msg = ActionSchema(action=action_value)
        key = ("train_001", 0, 1, action_value)
        memory.add(key, action_msg)
        retrieved = memory.get(key)

        assert isinstance(retrieved, ActionSchema)
        assert retrieved.action == action_value


def test_memory_hierarchical_keys():
    """Test that hierarchical key structure works correctly."""
    memory = Memory(capacity=10)
    message = EnvironmentSchema(done=False, reward=0.0, data={})

    key1 = ("train_001", 0, 1, 10)
    key2 = ("train_001", 0, 2, 10)
    key3 = ("train_001", 1, 1, 10)
    key4 = ("train_002", 0, 1, 10)

    memory.add(key1, message)
    memory.add(key2, message)
    memory.add(key3, message)
    memory.add(key4, message)

    assert len(memory) == 4
    assert all(key in memory for key in [key1, key2, key3, key4])


def test_memory_large_capacity():
    """Test that memory works with large capacity values."""
    memory = Memory(capacity=1000000)

    assert memory.capacity == 1000000
    assert len(memory) == 0


def test_memory_capacity_one(sample_environment_message):
    """Test that memory works correctly with capacity of 1."""
    memory = Memory(capacity=1)
    key1 = ("train_001", 0, 1, 1)
    key2 = ("train_001", 0, 1, 2)

    memory.add(key1, sample_environment_message)
    memory.add(key2, sample_environment_message)

    assert len(memory) == 1
    assert key1 not in memory
    assert key2 in memory


def test_memory_episode_boundary():
    """Test memory behavior across episode boundaries."""
    memory = Memory(capacity=10)

    # Add messages from episode 1
    for step in range(3):
        msg = EnvironmentSchema(done=(step == 2), reward=float(step), data={})
        memory.add(("train_001", 0, 1, step), msg)

    # Add messages from episode 2
    for step in range(3):
        msg = EnvironmentSchema(done=(step == 2), reward=float(step + 10), data={})
        memory.add(("train_001", 0, 2, step), msg)

    # Verify both episodes are stored
    assert ("train_001", 0, 1, 0) in memory
    assert ("train_001", 0, 2, 0) in memory

    # Verify terminal state is marked
    terminal_msg = memory.get(("train_001", 0, 1, 2))
    assert terminal_msg.done is True


def test_memory_parallel_workers():
    """Test memory handles parallel running instances correctly."""
    memory = Memory(capacity=20)

    # Simulate 3 parallel workers at same step
    for worker_id in range(3):
        msg = EnvironmentSchema(
            done=False, reward=float(worker_id), data={"worker_id": worker_id}
        )
        memory.add(("train_001", worker_id, 1, 10), msg)

    # Verify all workers stored independently
    for worker_id in range(3):
        retrieved = memory.get(("train_001", worker_id, 1, 10))
        assert retrieved.data["worker_id"] == worker_id
