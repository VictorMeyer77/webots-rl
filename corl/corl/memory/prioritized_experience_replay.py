import numpy as np
from numpy.typing import NDArray


class SumTree:
    """
    Binary sum tree for O(log n) prioritized sampling.

    Each leaf stores the priority of one experience. Internal nodes store the
    sum of their subtree, so the root always holds the total priority sum.
    Experiences are stored in a fixed-size circular buffer — when full, the
    oldest entry is overwritten.

    Tree layout for ``capacity=4`` (indices into ``self.tree``)::

                    0  (root = total sum)
                   / \\
                  1    2
                 / \\  / \\
                3   4 5   6   ← leaves (experiences)

    Attributes:
        capacity: Maximum number of experiences the tree can hold.
        tree: Float array of length ``2 * capacity - 1`` storing priority
            sums. Leaves occupy indices ``capacity - 1`` to
            ``2 * capacity - 2``.
        data: Object array of length ``capacity`` storing the raw experience
            tuples at each leaf position.
        write: Next write position in the circular buffer (``0`` to
            ``capacity - 1``).
        n_entries: Number of valid experiences currently stored
            (``0`` to ``capacity``).
    """

    def __init__(self, capacity: int):
        """
        Initialise the SumTree with the given capacity.

        Args:
            capacity: Maximum number of experiences to store.

        Raises:
            ValueError: If ``capacity`` is less than 1.
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    @property
    def total(self) -> float:
        """
        Get the total sum of all priorities in the tree.

        Returns:
            float: Sum of all priorities (value at root node, index 0).
        """
        return self.tree[0]

    def add(self, priority: float, data: tuple):
        """
        Add an experience to the tree at the current write position.

        Stores the data in the circular buffer and updates the corresponding
        leaf priority, propagating the change upward to maintain the sum
        invariant. When the buffer is full, the oldest entry is overwritten.

        Args:
            priority: Priority value for this experience (typically
                ``|δ|^alpha`` where ``δ`` is the TD-error).
            data: Experience tuple
                ``(state, action, reward, next_state, done)``.
        """
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float):
        """
        Update the priority at ``idx`` and propagate the change to the root.

        Computes the delta between the new and old priority, applies it to
        the node, then walks up the tree adding the delta to each ancestor.
        Time complexity: O(log n).

        Args:
            idx: Index in ``self.tree`` to update. Must be a leaf index
                (i.e. ``>= capacity - 1``).
            priority: New priority value to assign to this node.
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority

        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s: float):
        """
        Retrieve the leaf corresponding to cumulative priority value ``s``.

        Performs a top-down tree walk: at each internal node, go left if
        ``s`` does not exceed the left child's subtree sum, otherwise
        subtract the left sum and go right. Time complexity: O(log n).

        Args:
            s: Cumulative priority value in ``[0, self.total)``. Typically
                drawn as ``np.random.uniform(0, self.total)``.

        Returns:
            tuple[int, float, Any]: ``(tree_index, priority, data)`` where

            - ``tree_index`` is the leaf's index in ``self.tree`` (pass
              to :meth:`update` to revise its priority later).
            - ``priority`` is the leaf's current priority value.
            - ``data`` is the stored experience tuple
              ``(state, action, reward, next_state, done)``.
        """
        idx = 0

        # search down the tree
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1

            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedExperienceReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer (Schaul et al., 2015).

    Samples experiences proportionally to their TD-error priority rather
    than uniformly, so high-surprise transitions are revisited more often.
    A :class:`SumTree` provides O(log n) insertion, sampling, and priority
    updates. Importance-sampling (IS) weights are returned alongside each
    batch to correct the bias introduced by non-uniform sampling.

    Attributes:
        tree: Underlying :class:`SumTree` storing priorities and experiences.
        alpha: Priority exponent in ``[0, 1]``. ``0`` = uniform sampling,
            ``1`` = full prioritisation. Typical value: ``0.6``.
        max_priority: Highest priority seen so far. New experiences are
            assigned this value so they are sampled at least once before
            their TD-error is known.
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        Initialise the PER buffer.

        Args:
            capacity: Maximum number of experiences to store.
            alpha: Priority exponent. ``0`` disables prioritisation
                (uniform sampling); ``1`` uses full TD-error priority.
                Defaults to ``0.6``.
        """
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0

    def add(
        self,
        state: NDArray[np.float32],
        action: int,
        reward: float,
        next_state: NDArray[np.float32],
        done: bool,
    ) -> None:
        """
        Store a transition in the buffer with maximum priority.

        New experiences receive ``max_priority ** alpha`` so they are
        guaranteed to be sampled at least once. Their priority is revised
        to reflect the actual TD-error after the first training step via
        :meth:`update_priorities`.

        Args:
            state: Current observation array.
            action: Action taken in ``state``.
            reward: Scalar reward received after ``action``.
            next_state: Observation following ``action``.
            done: ``True`` if the episode terminated after this transition.
        """
        priority = self.max_priority**self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.int32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.bool_],
        list[int],
        NDArray[np.float32],
    ]:
        """
        Sample a batch of experiences using stratified prioritised sampling.

        The total priority range is divided into ``batch_size`` equal
        segments. One experience is drawn from each segment proportionally
        to priority, ensuring diversity. Importance-sampling weights
        ``w_i = (N · P(i))^{-β}`` are computed and normalised by the
        maximum weight so all values lie in ``(0, 1]``.

        Args:
            batch_size: Number of experiences to sample.
            beta: IS correction exponent in ``[0, 1]``. ``0`` applies no
                correction; ``1`` gives fully unbiased updates. Typically
                annealed from ``0.4`` → ``1.0`` over training.
                Defaults to ``0.4``.

        Returns:
            A 7-tuple ``(states, actions, rewards, next_states, dones,
            idxs, weights)`` where:

            - ``states`` — shape ``(batch_size, obs_dim)``, dtype float32
            - ``actions`` — shape ``(batch_size,)``, dtype int32
            - ``rewards`` — shape ``(batch_size,)``, dtype float32
            - ``next_states`` — shape ``(batch_size, obs_dim)``, dtype float32
            - ``dones`` — shape ``(batch_size,)``, dtype bool
            - ``idxs`` — list of tree indices; pass to
              :meth:`update_priorities` after computing new TD-errors
            - ``weights`` — IS weights, shape ``(batch_size,)``, dtype
              float32, normalised to ``[0, 1]``
        """
        batch = []
        idxs = []
        priorities = []

        segment = self.tree.total / batch_size

        for i in range(batch_size):
            s = np.random.uniform(
                segment * i, min(segment * (i + 1), self.tree.total - 1e-6)
            )
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        states, actions, rewards, next_states, dones = zip(*batch)

        sampling_prob = np.array(priorities) / self.tree.total
        weights = (len(self) * sampling_prob) ** (-beta)
        weights /= weights.max()

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            idxs,
            weights.astype(np.float32),
        )

    def update_priorities(self, idxs: list, td_errors: NDArray[np.float32]) -> None:
        """
        Revise experience priorities based on new TD-errors.

        Call this after each training step. Priorities are set to
        ``(|δ| + ε)^alpha`` where ``ε = 1e-6`` prevents any experience
        from having zero priority. :attr:`max_priority` is updated if any
        new priority exceeds the current maximum.

        Args:
            idxs: Tree indices returned by :meth:`sample`. Identifies
                which experiences to update.
            td_errors: TD-errors for each sampled experience, shape
                ``(batch_size,)``. Signs are ignored; only magnitudes
                are used.
        """
        td_errors = np.abs(td_errors) + 1e-6
        priorities = td_errors**self.alpha

        for idx, p in zip(idxs, priorities):
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self) -> int:
        """
        Return the number of experiences currently stored.

        Returns:
            int: Value in ``[0, capacity]``. O(1) — delegates to
            :attr:`SumTree.n_entries`.
        """
        return self.tree.n_entries
