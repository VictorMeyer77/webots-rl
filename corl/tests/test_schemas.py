"""
Unit tests for tracker schemas module.

Tests cover:
- StepKey initialization and validation
- StepKey hashing and comparison
- StepKey immutability (frozen model)
- StepResult initialization with various field types
- StepResult is_complete() method logic
- Field validation and constraints
- Type handling for numpy arrays
"""

import numpy as np
import pytest
from pydantic import ValidationError

from corl.schemas.tracker import StepKey, StepResult


class TestStepKey:
    """Test StepKey dataclass."""

    def test_initialization_valid(self):
        """Test initialization with valid values."""
        key = StepKey(worker_id=0, episode_id=1, step=2)
        assert key.worker_id == 0
        assert key.episode_id == 1
        assert key.step == 2

    def test_initialization_all_zeros(self):
        """Test initialization with all zeros (boundary case)."""
        key = StepKey(worker_id=0, episode_id=0, step=0)
        assert key.worker_id == 0
        assert key.episode_id == 0
        assert key.step == 0

    def test_initialization_large_values(self):
        """Test initialization with large values."""
        key = StepKey(worker_id=999999, episode_id=1000000, step=50000)
        assert key.worker_id == 999999
        assert key.episode_id == 1000000
        assert key.step == 50000

    def test_negative_worker_id(self):
        """Test that negative worker_id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            StepKey(worker_id=-1, episode_id=0, step=0)
        assert "worker_id" in str(exc_info.value)

    def test_negative_episode_id(self):
        """Test that negative episode_id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            StepKey(worker_id=0, episode_id=-1, step=0)
        assert "episode_id" in str(exc_info.value)

    def test_negative_step(self):
        """Test that negative step raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            StepKey(worker_id=0, episode_id=0, step=-1)
        assert "step" in str(exc_info.value)

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            StepKey(worker_id=0, episode_id=0)  # Missing step

        with pytest.raises(ValidationError):
            StepKey(worker_id=0)  # Missing episode_id and step

        with pytest.raises(ValidationError):
            StepKey()  # Missing all fields

    def test_hash_consistency(self):
        """Test that identical StepKeys have the same hash."""
        key1 = StepKey(worker_id=1, episode_id=2, step=3)
        key2 = StepKey(worker_id=1, episode_id=2, step=3)
        assert hash(key1) == hash(key2)

    def test_hash_different_values(self):
        """Test that different StepKeys have different hashes."""
        key1 = StepKey(worker_id=1, episode_id=2, step=3)
        key2 = StepKey(worker_id=1, episode_id=2, step=4)
        key3 = StepKey(worker_id=2, episode_id=2, step=3)

        # Note: Hash collisions are possible but unlikely for these values
        assert hash(key1) != hash(key2)
        assert hash(key1) != hash(key3)

    def test_hash_usable_in_dict(self):
        """Test that StepKey can be used as dictionary key."""
        key1 = StepKey(worker_id=1, episode_id=2, step=3)
        key2 = StepKey(worker_id=1, episode_id=2, step=3)
        key3 = StepKey(worker_id=2, episode_id=2, step=3)

        test_dict = {key1: "value1", key3: "value3"}
        assert test_dict[key1] == "value1"
        assert test_dict[key2] == "value1"  # Same as key1
        assert test_dict[key3] == "value3"

    def test_hash_usable_in_set(self):
        """Test that StepKey can be used in sets."""
        key1 = StepKey(worker_id=1, episode_id=2, step=3)
        key2 = StepKey(worker_id=1, episode_id=2, step=3)
        key3 = StepKey(worker_id=2, episode_id=2, step=3)

        test_set = {key1, key2, key3}
        assert len(test_set) == 2  # key1 and key2 are identical
        assert key1 in test_set
        assert key2 in test_set
        assert key3 in test_set

    def test_less_than_comparison(self):
        """Test __lt__ comparison method."""
        key1 = StepKey(worker_id=0, episode_id=0, step=0)
        key2 = StepKey(worker_id=0, episode_id=0, step=1)
        key3 = StepKey(worker_id=0, episode_id=1, step=0)
        key4 = StepKey(worker_id=1, episode_id=0, step=0)

        # Compare by step when worker_id and episode_id are same
        assert key1 < key2
        assert not key2 < key1

        # Compare by episode_id when worker_id is same
        assert key1 < key3
        assert not key3 < key1

        # Compare by worker_id first
        assert key1 < key4
        assert not key4 < key1

    def test_less_than_comparison_complex(self):
        """Test __lt__ comparison with more complex scenarios."""
        key1 = StepKey(worker_id=1, episode_id=2, step=3)
        key2 = StepKey(worker_id=1, episode_id=2, step=3)
        key3 = StepKey(worker_id=2, episode_id=1, step=1)

        # Equal keys
        assert not key1 < key2
        assert not key2 < key1

        # worker_id takes precedence even if other fields are smaller
        assert key1 < key3
        assert not key3 < key1

    def test_sorting(self):
        """Test that StepKeys can be sorted."""
        keys = [
            StepKey(worker_id=2, episode_id=1, step=1),
            StepKey(worker_id=1, episode_id=3, step=0),
            StepKey(worker_id=1, episode_id=2, step=5),
            StepKey(worker_id=1, episode_id=2, step=3),
        ]

        sorted_keys = sorted(keys)

        # Should be sorted by worker_id, then episode_id, then step
        assert sorted_keys[0].worker_id == 1
        assert sorted_keys[1].worker_id == 1
        assert sorted_keys[2].worker_id == 1
        assert sorted_keys[3].worker_id == 2

        # Within worker_id=1, check episode_id sorting
        assert sorted_keys[0].episode_id == 2
        assert sorted_keys[1].episode_id == 2
        assert sorted_keys[2].episode_id == 3

        # Within worker_id=1, episode_id=2, check step sorting
        assert sorted_keys[0].step == 3
        assert sorted_keys[1].step == 5

    def test_immutability(self):
        """Test that StepKey is frozen (immutable)."""
        key = StepKey(worker_id=1, episode_id=2, step=3)

        with pytest.raises(ValidationError):
            key.worker_id = 5

        with pytest.raises(ValidationError):
            key.episode_id = 10

        with pytest.raises(ValidationError):
            key.step = 20

    def test_equality(self):
        """Test equality comparison between StepKeys."""
        key1 = StepKey(worker_id=1, episode_id=2, step=3)
        key2 = StepKey(worker_id=1, episode_id=2, step=3)
        key3 = StepKey(worker_id=1, episode_id=2, step=4)

        assert key1 == key2
        assert not key1 == key3
        assert key1 != key3


class TestStepResult:
    """Test StepResult dataclass."""

    def test_initialization_all_none(self):
        """Test initialization with all default None values."""
        result = StepResult()
        assert result.observation is None
        assert result.action is None
        assert result.reward is None
        assert result.done is None

    def test_initialization_with_values(self):
        """Test initialization with all fields provided."""
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = StepResult(observation=obs, action=[2.0], reward=1.5, done=False)

        assert np.array_equal(result.observation, obs)
        assert result.action == [2.0]
        assert result.reward == 1.5
        assert result.done is False

    def test_initialization_partial(self):
        """Test initialization with some fields provided."""
        result = StepResult(action=[1.0], reward=0.5)
        assert result.observation is None
        assert result.action == [1.0]
        assert result.reward == 0.5
        assert result.done is None

    def test_observation_numpy_array(self):
        """Test observation field with numpy array."""
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = StepResult(observation=obs)

        assert isinstance(result.observation, np.ndarray)
        assert result.observation.dtype == np.float32
        assert np.array_equal(result.observation, obs)

    def test_observation_multidimensional(self):
        """Test observation field with multidimensional numpy array."""
        obs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = StepResult(observation=obs)

        assert result.observation.shape == (2, 2)
        assert np.array_equal(result.observation, obs)

    def test_observation_large_array(self):
        """Test observation field with large numpy array."""
        obs = np.random.randn(100, 100).astype(np.float32)
        result = StepResult(observation=obs)

        assert result.observation.shape == (100, 100)
        assert np.array_equal(result.observation, obs)

    def test_action_integer(self):
        """Test action field with integer value."""
        result = StepResult(action=[5.0])
        assert result.action == [5.0]
        assert isinstance(result.action, list)

    def test_action_zero(self):
        """Test action field with zero value."""
        result = StepResult(action=[0.0])
        assert result.action == [0.0]

    def test_reward_positive(self):
        """Test reward field with positive value."""
        result = StepResult(reward=10.5)
        assert result.reward == 10.5

    def test_reward_negative(self):
        """Test reward field with negative value."""
        result = StepResult(reward=-5.2)
        assert result.reward == -5.2

    def test_reward_zero(self):
        """Test reward field with zero value."""
        result = StepResult(reward=0.0)
        assert result.reward == 0.0

    def test_reward_integer(self):
        """Test reward field with integer value (should be converted to float)."""
        result = StepResult(reward=1)
        assert result.reward == 1.0
        assert isinstance(result.reward, (int, float))

    def test_done_true(self):
        """Test done field with True value."""
        result = StepResult(done=True)
        assert result.done is True

    def test_done_false(self):
        """Test done field with False value."""
        result = StepResult(done=False)
        assert result.done is False

    def test_is_complete_all_none(self):
        """Test is_complete() returns False when all fields are None."""
        result = StepResult()
        assert result.is_complete() is False

    def test_is_complete_all_filled(self):
        """Test is_complete() returns True when all fields are filled."""
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = StepResult(observation=obs, action=[2.0], reward=1.5, done=False)
        assert result.is_complete() is True

    def test_is_complete_missing_observation(self):
        """Test is_complete() returns False when observation is missing."""
        result = StepResult(action=[2.0], reward=1.5, done=False)
        assert result.is_complete() is False

    def test_is_complete_missing_action(self):
        """Test is_complete() returns False when action is missing."""
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = StepResult(observation=obs, reward=1.5, done=False)
        assert result.is_complete() is False

    def test_is_complete_missing_reward(self):
        """Test is_complete() returns False when reward is missing."""
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = StepResult(observation=obs, action=[2.0], done=False)
        assert result.is_complete() is False

    def test_is_complete_missing_done(self):
        """Test is_complete() returns False when done is missing."""
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = StepResult(observation=obs, action=[2.0], reward=1.5)
        assert result.is_complete() is False

    def test_is_complete_partial_filled(self):
        """Test is_complete() returns False when only some fields are filled."""
        result = StepResult(action=[2.0], reward=1.5)
        assert result.is_complete() is False

    def test_is_complete_done_true(self):
        """Test is_complete() with terminal state (done=True)."""
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = StepResult(observation=obs, action=[2.0], reward=1.5, done=True)
        assert result.is_complete() is True
        assert result.done is True

    def test_mutability(self):
        """Test that StepResult fields can be modified (not frozen)."""
        result = StepResult()

        # Should be able to set fields after initialization
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result.observation = obs
        result.action = [2.0]
        result.reward = 1.5
        result.done = False

        assert np.array_equal(result.observation, obs)
        assert result.action == [2.0]
        assert result.reward == 1.5
        assert result.done is False

    def test_observation_none_explicit(self):
        """Test explicitly setting observation to None."""
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = StepResult(observation=obs)
        result.observation = None
        assert result.observation is None

    def test_action_none_explicit(self):
        """Test explicitly setting action to None."""
        result = StepResult(action=[5.0])
        result.action = None
        assert result.action is None

    def test_reward_none_explicit(self):
        """Test explicitly setting reward to None."""
        result = StepResult(reward=1.5)
        result.reward = None
        assert result.reward is None

    def test_done_none_explicit(self):
        """Test explicitly setting done to None."""
        result = StepResult(done=True)
        result.done = None
        assert result.done is None

    def test_multiple_step_results(self):
        """Test creating multiple StepResult instances."""
        results = []
        for i in range(5):
            obs = np.array([float(i)], dtype=np.float32)
            result = StepResult(
                observation=obs, action=[float(i)], reward=float(i) * 0.1, done=(i == 4)
            )
            results.append(result)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.observation[0] == float(i)
            assert result.action == [float(i)]
            assert result.reward == float(i) * 0.1
            assert result.done == (i == 4)

    def test_reward_extreme_values(self):
        """Test reward field with extreme values."""
        result1 = StepResult(reward=1e10)
        assert result1.reward == 1e10

        result2 = StepResult(reward=-1e10)
        assert result2.reward == -1e10

        result3 = StepResult(reward=1e-10)
        assert result3.reward == 1e-10
