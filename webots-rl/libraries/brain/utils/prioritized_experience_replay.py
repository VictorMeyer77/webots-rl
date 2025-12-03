"""
Prioritized Experience Replay (PER) implementation for Deep Reinforcement Learning.

This module provides data structures and algorithms for Prioritized Experience Replay,
a technique that improves sample efficiency in Deep Q-Learning by sampling experiences
based on their temporal difference (TD) errors rather than uniformly.

Key Concepts:
    - **TD-Error Priority**: Experiences with higher TD-errors (surprises) are sampled
      more frequently, allowing the agent to learn more from important transitions.
    - **Importance Sampling**: Corrects bias introduced by non-uniform sampling using
      annealed importance sampling weights (beta: 0.4 → 1.0).
    - **SumTree Data Structure**: Enables O(log n) sampling and priority updates using
      a binary heap representation.
    - **Stratified Sampling**: Divides priority range into segments for batch diversity.

Classes:
    SumTree: Binary sum tree for efficient O(log n) proportional prioritized sampling.
        Stores priorities in internal nodes (sums) and experiences in leaf nodes.

    PrioritizedExperienceReplayBuffer: Complete PER buffer implementation combining
        SumTree with importance sampling weights and priority management.

Algorithm Overview:
    1. **Add Experience**: New experiences assigned max priority to ensure sampling
    2. **Sample Batch**: Stratified sampling proportional to TD-error priorities
    3. **Compute IS Weights**: Calculate importance sampling weights for bias correction
    4. **Train Model**: Use sampled batch with IS weights applied to loss
    5. **Update Priorities**: Update priorities based on new TD-errors after training

Mathematical Formulation:
    Priority:
        P(i) = |δ_i|^α + ε
        where δ_i is TD-error, α controls prioritization (0=uniform, 1=full)

    Sampling Probability:
        p(i) = P(i) / Σ_k P(k)

    Importance Sampling Weight:
        w_i = (N * p(i))^(-β) / max_j w_j
        where β anneals from 0.4 to 1.0 during training
"""

import numpy as np


class SumTree:
    """
    Binary sum tree data structure for efficient prioritized sampling.

    Used in Prioritized Experience Replay (PER) to sample experiences based on their TD-error priorities.
    The tree stores priorities in a binary tree where:
      - Leaf nodes (indices capacity-1 to 2*capacity-2) store individual priorities
      - Internal nodes store the sum of their children's priorities
      - Root node (index 0) stores the total sum of all priorities

    Structure example for capacity=5:
                    root (total_sum)
                   /                \
              sum_left            sum_right
              /      \            /        \
          p[0]      p[1]      p[2]      p[3]    p[4]

    Attributes:
        capacity (int): Maximum number of experiences that can be stored.
        tree (np.ndarray): Array of size 2*capacity-1 representing the binary tree (stores priority sums).
        data (np.ndarray): Array of size capacity storing the actual experience data at leaf positions.
        write (int): Current write position for circular buffer (0 to capacity-1).
        n_entries (int): Current number of experiences stored in the tree (0 to capacity).
    """

    def __init__(self, capacity: int):
        """
        Initialize the SumTree with the given capacity.

        Args:
            capacity (int): Maximum number of experiences to store.
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
        Add a new experience with the given priority to the tree.

        The experience is added at the current write position (circular buffer behavior).
        The priority is stored in the tree and propagated upward to maintain sum invariants.
        When the buffer is full, the oldest experience is overwritten.

        Args:
            priority (float): Priority value for this experience (typically TD-error or |δ|).
            data (tuple): Experience tuple, e.g., (state, action, reward, next_state, done).

        Algorithm:
            1. Calculate tree index from data write position (leaf node = write + capacity - 1)
            2. Store experience data at current write position
            3. Update tree with new priority (propagates changes upward)
            4. Advance write pointer (wraps around when reaching capacity)
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
        Update the priority of a specific tree node and propagate changes upward.

        This maintains the sum-tree property where each parent node equals the sum
        of its children's priorities. The change propagates all the way to the root.

        Args:
            idx (int): Index in the tree array to update (typically a leaf node index).
            priority (float): New priority value to assign to this node.

        Algorithm:
            1. Calculate the difference between new and old priority
            2. Update the node's priority
            3. Traverse upward to root, updating each ancestor by adding the change
            4. Parent of node i is at index (i-1)//2

        Time Complexity: O(log n) - traverse height of tree
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority

        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s: float):
        """
        Retrieve an experience by sampling proportionally to priorities.

        This performs a top-down search through the tree to find the leaf node
        corresponding to a cumulative priority value s. Used for proportional
        prioritized sampling in Prioritized Experience Replay.

        Args:
            s (float): Random value between 0 and self.total for proportional sampling.
                      Typically sampled as: random.uniform(0, self.total)

        Returns:
            tuple: (tree_index, priority, experience_data)
                - tree_index (int): Index in the tree array (used for future priority updates)
                - priority (float): The priority value of the selected experience
                - experience_data: The stored experience tuple (state, action, reward, next_state, done)

        Algorithm:
            1. Start at root node (idx=0)
            2. At each internal node, compare s with left child's sum:
               - If s <= left_sum: go to left child
               - Else: subtract left_sum from s and go to right child
            3. Repeat until reaching a leaf node (idx >= capacity-1)
            4. Convert tree index to data index and return

        Example:
            Given priorities [0.1, 0.2, 0.3] with total=0.6:
            - Sample s=0.15: returns experience with priority 0.2 (33.3% probability)
            - Sample s=0.05: returns experience with priority 0.1 (16.7% probability)

        Time Complexity: O(log n) - traverse height of tree
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
    Prioritized Experience Replay (PER) buffer for Deep Reinforcement Learning.

    Implements the Prioritized Experience Replay algorithm (Schaul et al., 2015) which
    samples experiences based on their TD-error priorities rather than uniformly.
    Experiences with higher TD-errors (surprises) are sampled more frequently, leading
    to more efficient learning.

    The buffer uses a SumTree for O(log n) sampling and updating operations.
    It also implements importance sampling weights to correct for the bias introduced
    by non-uniform sampling.

    Attributes:
        tree (SumTree): Binary sum tree for efficient priority-based sampling.
        alpha (float): Priority exponent controlling how much prioritization is used.
                      - alpha=0: uniform sampling (no prioritization)
                      - alpha=1: full prioritization based on TD-error
                      - typical values: 0.6-0.7
        max_priority (float): Maximum priority seen so far. New experiences are assigned
                             this priority to ensure they are sampled at least once.
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        Initialize the Prioritized Experience Replay buffer.

        Args:
            capacity (int): Maximum number of experiences to store in the buffer.
            alpha (float, optional): Priority exponent controlling prioritization strength.
                                    Defaults to 0.6. Range: [0, 1]
                                    - 0 = uniform sampling
                                    - 1 = full prioritization
        """
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Add a new experience transition to the replay buffer.

        New experiences are assigned the maximum priority seen so far to ensure
        they are sampled at least once. This prevents the agent from ignoring
        potentially important new experiences.

        Args:
            state (np.ndarray): Current state observation (e.g., sensor readings, image).
            action (int): Action taken in the current state.
            reward (float): Reward received after taking the action.
            next_state (np.ndarray): Next state observation after taking the action.
            done (bool): Whether the episode terminated after this transition.

        Note:
            The priority is calculated as max_priority^alpha, where alpha controls
            the degree of prioritization. The actual TD-error based priority will
            be updated later via update_priorities() after the experience is sampled
            and learned from.
        """
        priority = self.max_priority**self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> tuple[np.array, np.array, np.array, np.array, np.array, list[int], np.array]:
        """
        Sample a batch of experiences using prioritized sampling.

        The buffer is divided into batch_size segments, and one experience is sampled
        from each segment proportionally to priorities. This stratified sampling ensures
        diversity in the batch.

        Importance sampling weights are computed to correct for the bias introduced by
        non-uniform sampling. These weights should be used when updating the network
        (multiply the loss by the weight).

        Args:
            batch_size (int): Number of experiences to sample.
            beta (float, optional): Importance sampling exponent controlling bias correction.
                                   Defaults to 0.4. Range: [0, 1]
                                   - beta=0: no correction (biased updates)
                                   - beta=1: full correction (unbiased updates)
                                   - Typically annealed from 0.4 to 1.0 during training

        Returns:
            tuple: (states, actions, rewards, next_states, dones, idxs, weights)
                - states (np.ndarray): Batch of state observations, shape (batch_size, state_dim)
                - actions (np.ndarray): Batch of actions, shape (batch_size,)
                - rewards (np.ndarray): Batch of rewards, shape (batch_size,)
                - next_states (np.ndarray): Batch of next state observations, shape (batch_size, state_dim)
                - dones (np.ndarray): Batch of termination flags, shape (batch_size,)
                - idxs (list): Tree indices for updating priorities later
                - weights (np.ndarray): Importance sampling weights, shape (batch_size,)
                                       normalized to [0, 1] with max weight = 1

        Algorithm:
            1. Divide priority range into batch_size equal segments
            2. Sample one experience from each segment (stratified sampling)
            3. Compute importance sampling weights: w_i = (N * P(i))^(-beta)
            4. Normalize weights by dividing by max weight
        """
        batch = []
        idxs = []
        priorities = []

        segment = self.tree.total / batch_size

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
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
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=bool),
            idxs,
            weights.astype(np.float32),
        )

    def update_priorities(self, idxs: list, td_errors: np.ndarray) -> None:
        """
        Update the priorities of sampled experiences based on their TD-errors.

        This method should be called after training on a batch of experiences.
        The priorities are updated based on the absolute TD-errors, which measure
        how surprising or unexpected each transition was. Experiences with higher
        TD-errors will be sampled more frequently in future batches.

        Args:
            idxs (list): List of tree indices returned by sample() method.
                        These identify which experiences to update.
            td_errors (np.ndarray): Array of TD-errors (temporal difference errors) for each
                                 sampled experience. Shape: (batch_size,)
                                 TD-error = |target - prediction|

        Algorithm:
            1. Add small epsilon (1e-6) to TD-errors to ensure non-zero priorities
            2. Compute priorities as |TD-error|^alpha
            3. Update each experience's priority in the SumTree
            4. Track the maximum priority for assigning to new experiences

        Note:
            The epsilon value prevents priorities from becoming exactly zero,
            ensuring all experiences have a chance to be sampled.
        """
        td_errors = np.abs(td_errors) + 1e-6
        priorities = td_errors**self.alpha

        for idx, p in zip(idxs, priorities):
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self) -> int:
        """
        Get the current number of experiences stored in the buffer.

        Returns:
            int: Number of non-empty experiences currently in the buffer.
                 This will be less than or equal to capacity.

        Note:
            Uses the n_entries counter for O(1) time complexity.
        """
        return self.tree.n_entries
