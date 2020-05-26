from threading import Lock
import numpy as np


class SumTree:
    """ Interface """
    def __init__(self, capacity):
        self._capacity = capacity
        self._tree_size = capacity - 1

        self._container = np.zeros(int(self._tree_size + self._capacity))

        # locker used to avoid conflict introduced by tf.data.Dataset
        # ensuring SumTree update will not happen while sampling
        # which may cause out-of-range sampling in data_structure.find
        self._locker = Lock()

    @property
    def total_priorities(self):
        return self._container[0]

    def find(self, value):
        idx = 0                 # start from the root
        with self._locker:
            while idx < self._tree_size:
                left, right = 2 * idx + 1, 2 * idx + 2
                if value <= self._container[left]:
                    idx = left
                else:
                    idx = right
                    value -= self._container[left]

            return self._container[idx], idx - self._tree_size

    def batch_find(self, values):
        """ vectorized find """
        idxes = np.zeros_like(values, dtype=np.int32)
        with self._locker:
            while np.any(idxes < self._tree_size):
                left, right = 2 * idxes + 1, 2 * idxes + 2
                left = np.where(left < len(self._container), left, idxes)
                right = np.where(right < len(self._container), right, idxes)
                idxes = np.where(values <= self._container[left], left, right)
                values = np.where(values <= self._container[left], values, values - self._container[left])

            idx = np.zeros_like(values, dtype=np.int32)
            while np.all(idx < self._tree_size):
                left, right = 2 * idx + 1, 2 * idx + 2
                np.testing.assert_allclose(self._container[idx], self._container[left] + self._container[right],
                        err_msg=f'idx: {idx}\n{self._container[idx]}\nleft: {left}\n{self._container[left]}\nright: {self._container[right]}')
                idx = np.where(np.random.uniform(size=idx.shape) < .5, left, right)

            return self._container[idxes], idxes - self._tree_size

    def update(self, mem_idx, priority):
        np.testing.assert_array_less(0, priority)
        idx = mem_idx + self._tree_size
        with self._locker:
            diff = priority - self._container[idx]
            self._container[idx] += diff

            while idx > 0:
                idx = (idx - 1) // 2    # update idx to its parent idx
                self._container[idx] += diff
                left, right = 2 * idx + 1, 2 * idx + 2
                np.testing.assert_allclose(self._container[idx], self._container[left] + self._container[right],
                    err_msg=f'{idx}\n{self._container[idx]}\n{self._container[left]}\n{self._container[right]}')
            
    def batch_update(self, mem_idxes, priorities):
        """ vectorized update """
        np.testing.assert_array_less(0, priorities)
        idxes = mem_idxes + self._tree_size

        with self._locker:
            diffs = priorities - self._container[idxes]
            # the following two lines avoid error caused by duplicates in idxes
            # they almost equal to the following code, 
            # for i, d in zip(idxes, diffs):
            #     self._container[i] += d
            # except that they retain the unique idxes and diffs for later use 
            diffs = np.bincount(idxes, weights=diffs)
            idxes = np.arange(diffs.size)
            self._container[idxes] += diffs

            while len(idxes) > 0:
                p_idxes = (idxes - 1) // 2  # parent idxes
                idxes, idx1, count = np.unique(p_idxes, return_index=True, return_counts=True)
                
                _, idx2 = np.unique(p_idxes[::-1], return_index=True)
                # the following code only works for binary trees
                diffs = (diffs[-idx2-1] + diffs[idx1]) * count / 2
                np.testing.assert_equal(len(idxes), len(diffs))

                # code for unbalanced binary tree to avoid negative idxes.
                diffs = diffs[idxes >= 0]
                idxes = idxes[idxes >= 0]
                self._container[idxes] += diffs

            idxes = mem_idxes + self._tree_size
            while np.all(idxes > 0):
                idxes = (idxes - 1) // 2
                left, right = 2 * idxes + 1, 2 * idxes + 2
                np.testing.assert_allclose(self._container[idxes], self._container[left] + self._container[right],
                        err_msg=f'idx: {idxes}\n{self._container[idxes]}\nleft: {left}\n{self._container[left]}\nright: {self._container[right]}')
            