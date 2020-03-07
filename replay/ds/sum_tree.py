import numpy as np


class SumTree:
    """ Interface """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_size = capacity - 1

        self.container = np.zeros(self.tree_size + self.capacity)

    @property
    def total_priorities(self):
        return self.container[0]

    def find(self, value):
        idx = 0                 # start from the root

        while idx < self.tree_size:
            left, right = 2 * idx + 1, 2 * idx + 2
            if value <= self.container[left]:
                idx = left
            else:
                idx = right
                value -= self.container[left]

        return self.container[idx], idx - self.tree_size

    def batch_find(self, values):
        """vectorized batch find"""
        idxes = np.zeros_like(values, dtype=np.int32)

        while np.any(idxes < self.tree_size):
            left, right = 2 * idxes + 1, 2 * idxes + 2
            left = np.where(left < len(self.container), left, idxes)
            right = np.where(right < len(self.container), right, idxes)
            idxes = np.where(values <= self.container[left], left, right)
            values = np.where(values <= self.container[left], values, values - self.container[left])

        return self.container[idxes], idxes - self.tree_size

    def update(self, mem_idx, priority):
        idx = mem_idx + self.tree_size
        diff = self._compute_differences(idx, priority)
        self.container[idx] += diff

        while idx > 0:
            idx = (idx - 1) // 2    # update idx to its parent idx

            self.container[idx] += diff
            
    def batch_update(self, mem_idxes, priorities):
        """ vectorized batch update """
        idxes = mem_idxes + self.tree_size
        diffs = self._compute_differences(idxes, priorities)
        self.container[idxes] += diffs

        while len(idxes) > 0:
            p_idxes = (idxes - 1) // 2  # parent indexes
            idxes, idx1, count = np.unique(p_idxes, return_index=True, return_counts=True)

            _, idx2 = np.unique(p_idxes[::-1], return_index=True)
            # the following code only works for binary trees
            diffs = (diffs[-idx2-1] + diffs[idx1]) * count / 2
            np.testing.assert_equal(len(idxes), len(diffs))

            # code for unbalanced binary tree to avoid negative indexes.
            diffs = diffs[idxes >= 0]
            idxes = idxes[idxes >= 0]
            self.container[idxes] += diffs

    def _compute_differences(self, idxes, new_priorities):
        old_priorities = self.container[idxes]
        return new_priorities - old_priorities
