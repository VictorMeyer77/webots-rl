"""
Unit tests for corl.memory.prioritized_experience_replay
"""

import numpy as np
import pytest

from corl.memory.prioritized_experience_replay import (
    PrioritizedExperienceReplayBuffer,
    SumTree,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STATE_DIM = 4


def _experience(reward: float = 0.0, done: bool = False):
    state = np.zeros(STATE_DIM, dtype=np.float32)
    next_state = np.ones(STATE_DIM, dtype=np.float32)
    return state, 0, reward, next_state, done


# ---------------------------------------------------------------------------
# SumTree tests
# ---------------------------------------------------------------------------


class TestSumTreeInit:
    def test_capacity_stored(self):
        tree = SumTree(8)
        assert tree.capacity == 8

    def test_tree_array_size(self):
        tree = SumTree(8)
        assert len(tree.tree) == 2 * 8 - 1

    def test_data_array_size(self):
        tree = SumTree(8)
        assert len(tree.data) == 8

    def test_write_starts_at_zero(self):
        tree = SumTree(8)
        assert tree.write == 0

    def test_n_entries_starts_at_zero(self):
        tree = SumTree(8)
        assert tree.n_entries == 0

    def test_tree_initially_all_zeros(self):
        tree = SumTree(8)
        assert np.all(tree.tree == 0.0)


class TestSumTreeTotal:
    def test_total_is_zero_when_empty(self):
        tree = SumTree(4)
        assert tree.total == 0.0

    def test_total_equals_root_after_add(self):
        tree = SumTree(4)
        tree.add(0.5, ("a",))
        assert tree.total == pytest.approx(0.5)

    def test_total_is_sum_of_all_priorities(self):
        tree = SumTree(4)
        tree.add(0.1, ("a",))
        tree.add(0.3, ("b",))
        tree.add(0.6, ("c",))
        assert tree.total == pytest.approx(1.0)


class TestSumTreeAdd:
    def test_add_increments_n_entries(self):
        tree = SumTree(4)
        tree.add(1.0, ("x",))
        assert tree.n_entries == 1

    def test_add_advances_write_pointer(self):
        tree = SumTree(4)
        tree.add(1.0, ("x",))
        assert tree.write == 1

    def test_add_stores_data(self):
        tree = SumTree(4)
        data = ("state", 1, 2.0, "next", False)
        tree.add(1.0, data)
        assert tree.data[0] == data

    def test_add_updates_leaf_priority(self):
        tree = SumTree(4)
        tree.add(0.7, ("x",))
        leaf_idx = 0 + tree.capacity - 1
        assert tree.tree[leaf_idx] == pytest.approx(0.7)

    def test_write_wraps_at_capacity(self):
        capacity = 3
        tree = SumTree(capacity)
        for i in range(capacity + 1):
            tree.add(1.0, (i,))
        assert tree.write == 1

    def test_n_entries_capped_at_capacity(self):
        capacity = 3
        tree = SumTree(capacity)
        for i in range(capacity + 5):
            tree.add(1.0, (i,))
        assert tree.n_entries == capacity

    def test_circular_overwrite_updates_total(self):
        tree = SumTree(2)
        tree.add(1.0, ("a",))
        tree.add(1.0, ("b",))
        tree.add(2.0, ("c",))  # overwrites slot 0
        assert tree.total == pytest.approx(3.0)

    def test_circular_overwrite_replaces_data(self):
        tree = SumTree(2)
        tree.add(1.0, ("a",))
        tree.add(1.0, ("b",))
        tree.add(2.0, ("c",))
        assert tree.data[0] == ("c",)


class TestSumTreeUpdate:
    def test_update_changes_leaf_value(self):
        tree = SumTree(4)
        tree.add(0.5, ("x",))
        leaf_idx = tree.capacity - 1
        tree.update(leaf_idx, 0.9)
        assert tree.tree[leaf_idx] == pytest.approx(0.9)

    def test_update_propagates_to_root(self):
        tree = SumTree(4)
        tree.add(0.5, ("x",))
        leaf_idx = tree.capacity - 1
        tree.update(leaf_idx, 0.9)
        assert tree.total == pytest.approx(0.9)

    def test_update_maintains_sum_invariant(self):
        tree = SumTree(4)
        tree.add(0.2, ("a",))
        tree.add(0.3, ("b",))
        tree.add(0.4, ("c",))
        # update first leaf
        leaf_idx = tree.capacity - 1
        tree.update(leaf_idx, 0.5)
        assert tree.total == pytest.approx(0.5 + 0.3 + 0.4)

    def test_update_decrease_reduces_total(self):
        tree = SumTree(4)
        tree.add(1.0, ("x",))
        leaf_idx = tree.capacity - 1
        tree.update(leaf_idx, 0.1)
        assert tree.total == pytest.approx(0.1)


class TestSumTreeGet:
    def test_get_returns_three_tuple(self):
        tree = SumTree(4)
        tree.add(1.0, ("x",))
        result = tree.get(0.5)
        assert len(result) == 3

    def test_get_returns_correct_data(self):
        tree = SumTree(4)
        data = ("state", 1, 2.0, "next", False)
        tree.add(1.0, data)
        _, _, retrieved = tree.get(0.5)
        assert retrieved == data

    def test_get_returns_correct_priority(self):
        tree = SumTree(4)
        tree.add(0.8, ("x",))
        _, priority, _ = tree.get(0.5)
        assert priority == pytest.approx(0.8)

    def test_get_returns_leaf_tree_index(self):
        tree = SumTree(4)
        tree.add(1.0, ("x",))
        idx, _, _ = tree.get(0.5)
        assert idx >= tree.capacity - 1

    def test_get_selects_proportionally(self):
        """Higher priority experience should be sampled more often."""
        tree = SumTree(2)
        tree.add(0.1, ("low",))
        tree.add(0.9, ("high",))

        counts = {"low": 0, "high": 0}
        rng = np.random.default_rng(0)
        for _ in range(1000):
            s = rng.uniform(0, tree.total)
            _, _, data = tree.get(s)
            counts[data[0]] += 1

        assert counts["high"] > counts["low"]

    def test_get_at_total_minus_epsilon_returns_last_leaf(self):
        """Sampling near the total should still return a valid leaf."""
        tree = SumTree(4)
        tree.add(0.25, ("a",))
        tree.add(0.25, ("b",))
        tree.add(0.25, ("c",))
        tree.add(0.25, ("d",))
        idx, priority, _ = tree.get(tree.total - 1e-8)
        assert priority >= 0.0
        assert idx >= tree.capacity - 1


# ---------------------------------------------------------------------------
# PrioritizedExperienceReplayBuffer tests
# ---------------------------------------------------------------------------


class TestPERBufferInit:
    def test_default_alpha(self):
        buf = PrioritizedExperienceReplayBuffer(capacity=10)
        assert buf.alpha == 0.6

    def test_custom_alpha(self):
        buf = PrioritizedExperienceReplayBuffer(capacity=10, alpha=1.0)
        assert buf.alpha == 1.0

    def test_max_priority_starts_at_one(self):
        buf = PrioritizedExperienceReplayBuffer(capacity=10)
        assert buf.max_priority == 1.0

    def test_len_starts_at_zero(self):
        buf = PrioritizedExperienceReplayBuffer(capacity=10)
        assert len(buf) == 0


class TestPERBufferAdd:
    def test_add_increments_len(self):
        buf = PrioritizedExperienceReplayBuffer(capacity=10)
        buf.add(*_experience())
        assert len(buf) == 1

    def test_add_multiple_increments_len(self):
        buf = PrioritizedExperienceReplayBuffer(capacity=10)
        for _ in range(5):
            buf.add(*_experience())
        assert len(buf) == 5

    def test_len_capped_at_capacity(self):
        buf = PrioritizedExperienceReplayBuffer(capacity=4)
        for _ in range(10):
            buf.add(*_experience())
        assert len(buf) == 4

    def test_new_experience_gets_max_priority(self):
        buf = PrioritizedExperienceReplayBuffer(capacity=10, alpha=1.0)
        buf.max_priority = 2.0
        buf.add(*_experience())
        # With alpha=1, priority = max_priority^1 = max_priority
        # The total should equal 2.0
        assert buf.tree.total == pytest.approx(2.0)


class TestPERBufferSample:
    @pytest.fixture
    def full_buffer(self):
        buf = PrioritizedExperienceReplayBuffer(capacity=20, alpha=0.6)
        for i in range(20):
            buf.add(*_experience(reward=float(i)))
        return buf

    def test_sample_returns_seven_elements(self, full_buffer):
        result = full_buffer.sample(batch_size=4)
        assert len(result) == 7

    def test_states_shape(self, full_buffer):
        states, *_ = full_buffer.sample(batch_size=4)
        assert states.shape == (4, STATE_DIM)

    def test_actions_shape(self, full_buffer):
        _, actions, *_ = full_buffer.sample(batch_size=4)
        assert actions.shape == (4,)

    def test_rewards_shape(self, full_buffer):
        _, _, rewards, *_ = full_buffer.sample(batch_size=4)
        assert rewards.shape == (4,)

    def test_next_states_shape(self, full_buffer):
        _, _, _, next_states, *_ = full_buffer.sample(batch_size=4)
        assert next_states.shape == (4, STATE_DIM)

    def test_dones_shape(self, full_buffer):
        _, _, _, _, dones, *_ = full_buffer.sample(batch_size=4)
        assert dones.shape == (4,)

    def test_idxs_length(self, full_buffer):
        *_, idxs, _ = full_buffer.sample(batch_size=4)
        assert len(idxs) == 4

    def test_weights_shape(self, full_buffer):
        *_, weights = full_buffer.sample(batch_size=4)
        assert weights.shape == (4,)

    def test_weights_in_zero_one(self, full_buffer):
        *_, weights = full_buffer.sample(batch_size=4)
        assert np.all(weights >= 0.0)
        assert np.all(weights <= 1.0)

    def test_max_weight_is_one(self, full_buffer):
        *_, weights = full_buffer.sample(batch_size=4)
        assert weights.max() == pytest.approx(1.0)

    def test_states_dtype(self, full_buffer):
        states, *_ = full_buffer.sample(batch_size=4)
        assert states.dtype == np.float32

    def test_actions_dtype(self, full_buffer):
        _, actions, *_ = full_buffer.sample(batch_size=4)
        assert actions.dtype == np.int32

    def test_dones_dtype(self, full_buffer):
        _, _, _, _, dones, *_ = full_buffer.sample(batch_size=4)
        assert dones.dtype == np.bool_

    def test_weights_dtype(self, full_buffer):
        *_, weights = full_buffer.sample(batch_size=4)
        assert weights.dtype == np.float32

    def test_higher_beta_increases_low_priority_weights(self):
        """With beta=1, IS correction is stronger — weights vary more."""
        buf = PrioritizedExperienceReplayBuffer(capacity=20, alpha=0.6)
        for i in range(20):
            buf.add(*_experience(reward=float(i)))
        # Update some priorities so they differ
        buf.update_priorities(
            list(range(buf.tree.capacity - 1, buf.tree.capacity - 1 + 10)),
            np.array([0.01] * 5 + [5.0] * 5, dtype=np.float32),
        )
        *_, w_low = buf.sample(batch_size=8, beta=0.0)
        *_, w_high = buf.sample(batch_size=8, beta=1.0)
        # Higher beta => larger spread of weights
        assert w_high.std() >= w_low.std() or w_high.std() == pytest.approx(
            w_low.std(), abs=1e-3
        )


class TestPERBufferUpdatePriorities:
    def test_update_raises_max_priority(self):
        buf = PrioritizedExperienceReplayBuffer(capacity=10, alpha=1.0)
        for _ in range(5):
            buf.add(*_experience())
        states, _, _, _, _, idxs, _ = buf.sample(batch_size=5)
        buf.update_priorities(idxs, np.array([10.0] * 5, dtype=np.float32))
        # With alpha=1: priority = |td| + eps = 10 + 1e-6 > 1.0
        assert buf.max_priority > 1.0

    def test_update_priorities_uses_abs_td_error(self):
        buf = PrioritizedExperienceReplayBuffer(capacity=10, alpha=1.0)
        for _ in range(5):
            buf.add(*_experience())
        _, _, _, _, _, idxs, _ = buf.sample(batch_size=5)
        buf.update_priorities(idxs, np.array([-3.0] * 5, dtype=np.float32))
        # |td| + eps = 3 + 1e-6; with alpha=1 priority = 3 + 1e-6
        assert buf.max_priority == pytest.approx(3.0 + 1e-6)

    def test_update_priorities_changes_tree_totals(self):
        buf = PrioritizedExperienceReplayBuffer(capacity=10, alpha=1.0)
        for _ in range(10):
            buf.add(*_experience())
        total_before = buf.tree.total
        _, _, _, _, _, idxs, _ = buf.sample(batch_size=5)
        buf.update_priorities(idxs, np.full(5, 99.0, dtype=np.float32))
        assert buf.tree.total != pytest.approx(total_before)

    def test_max_priority_never_decreases(self):
        buf = PrioritizedExperienceReplayBuffer(capacity=10, alpha=1.0)
        for _ in range(10):
            buf.add(*_experience())
        _, _, _, _, _, idxs, _ = buf.sample(batch_size=5)
        buf.update_priorities(idxs, np.full(5, 5.0, dtype=np.float32))
        high_max = buf.max_priority
        _, _, _, _, _, idxs2, _ = buf.sample(batch_size=5)
        buf.update_priorities(idxs2, np.full(5, 0.001, dtype=np.float32))
        assert buf.max_priority == pytest.approx(high_max)


class TestPERBufferLen:
    def test_len_empty(self):
        buf = PrioritizedExperienceReplayBuffer(capacity=10)
        assert len(buf) == 0

    def test_len_partial_fill(self):
        buf = PrioritizedExperienceReplayBuffer(capacity=10)
        for _ in range(7):
            buf.add(*_experience())
        assert len(buf) == 7

    def test_len_full(self):
        buf = PrioritizedExperienceReplayBuffer(capacity=5)
        for _ in range(5):
            buf.add(*_experience())
        assert len(buf) == 5

    def test_len_overflow(self):
        buf = PrioritizedExperienceReplayBuffer(capacity=5)
        for _ in range(8):
            buf.add(*_experience())
        assert len(buf) == 5
