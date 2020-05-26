import numpy as np

from core.decorator import override
from utility.schedule import PiecewiseSchedule
from replay.base import Replay
from replay.ds.sum_tree import SumTree


class PERBase(Replay):
    """ Base class for PER, left in case one day I implement rank-based PER """
    def __init__(self, config):
        super().__init__(config)
        self._data_structure = None            

        # params for prioritized replay
        self._beta = float(config.get('beta0', .4))
        if '_beta_steps' in config:
            self._beta_schedule = PiecewiseSchedule(
                [(0, self._beta), (self._beta_steps, 1.)], 
                outside_value=1.)

        self._top_priority = 1.

        self._sample_i = 0   # count how many times self._sample is called

    @override(Replay)
    def sample(self, batch_size=None):
        assert self.good_to_learn(), (
            'There are not sufficient transitions to start learning --- '
            f'transitions in buffer({len(self)}) vs '
            f'minimum required size({self._min_size})')
        samples = self._sample(batch_size=batch_size)
        self._sample_i += 1
        if hasattr(self, '_beta_schedule'):
            self._update_beta()
        return samples

    @override(Replay)
    def add(self, **kwargs):
        # it is okay to add when sampling as it does not decrease the priorities, 
        # so no locker is needed
        super().add(**kwargs)
        # super().add updates self._mem_idx 
        self._data_structure.update(self._mem_idx - 1, self._top_priority)

    def update_priorities(self, priorities, idxes):
        assert not np.any(np.isnan(priorities)), priorities
        np.testing.assert_array_less(0, priorities)
        if self._to_update_top_priority:
            self._top_priority = max(self._top_priority, np.max(priorities))
        self._data_structure.batch_update(idxes, priorities)
        # for i, p in zip(idxes, priorities):
        #     self._data_structure.update(i, p)

    """ Implementation """
    def _update_beta(self):
        self._beta = self._beta_schedule.value(self._sample_i)

    @override(Replay)
    def _merge(self, local_buffer, length):
        assert np.all(local_buffer['priority'][: length] != 0)
        # update sum tree
        mem_idxes = np.arange(self._mem_idx, self._mem_idx + length) % self._capacity
        np.testing.assert_equal(len(mem_idxes), len(local_buffer['priority']))
        self._data_structure.batch_update(mem_idxes, local_buffer['priority'])
        del local_buffer['priority']
        # update memory
        super()._merge(local_buffer, length)
        
    def _compute_IS_ratios(self, probabilities):
        IS_ratios = (np.min(probabilities) / probabilities)**self._beta

        return IS_ratios


class ProportionalPER(PERBase):
    """ Interface """
    def __init__(self, config):
        super().__init__(config)
        self._data_structure = SumTree(self._capacity)        # mem_idx    -->     priority

    """ Implementation """
    @override(PERBase)
    def _sample(self, batch_size=None):
        batch_size = batch_size or self._batch_size
        total_priorities = self._data_structure.total_priorities

        intervals = np.linspace(0, total_priorities, batch_size+1)
        # print('before', total_priorities, intervals)
        values = np.random.uniform(intervals[:-1], intervals[1:])
        priorities, idxes = self._data_structure.batch_find(values)
        assert np.max(idxes) < len(self), f'{idxes}\n{values}\n{priorities}\n{total_priorities}, {len(self)}'
        assert np.min(priorities) > 0, f'idxes: {idxes}\nvalues: {values}\npriorities: {priorities}\ntotal: {total_priorities}, len: {len(self)}'

        probabilities = priorities / total_priorities

        # compute importance sampling ratios
        IS_ratios = self._compute_IS_ratios(probabilities)
        samples = self._get_samples(idxes)
        samples['IS_ratio'] = IS_ratios
        samples['idxes'] = idxes

        return samples
