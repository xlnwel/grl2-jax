from threading import Lock
import numpy as np

from utility.decorators import override
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
        self._beta_schedule = PiecewiseSchedule([(0, self.beta), (float(config['beta_steps']), 1.)], 
                                                outside_value=1.)

        self._top_priority = 2.
        self._to_update_top_priority = config.get('to_update_top_priority')

        self._sample_i = 0   # count how many times self._sample is called

        # locker used to avoid conflict introduced by tf.data.Dataset
        # ensuring SumTree update will not happen while sampling
        # which may cause out-of-range sampling in data_structure.find
        self._locker = Lock()

    @override(Replay)
    def sample(self, batch_size=None):
        assert self.good_to_learn(), (
            'There are not sufficient transitions to start learning --- '
            f'transitions in buffer({len(self)}) vs '
            f'minimum required size({self.min_size})')
        with self._locker:
            samples = self._sample(batch_size=batch_size)
            self._sample_i += 1
            self._update_beta()

        return samples

    @override(Replay)
    def add(self, **kwargs):
        # it is okay to add when sampling, so no locker is needed
        super().add(**kwargs)
        # super().add updates self._mem_idx 
        self._data_structure.update(self._mem_idx - 1, self.top_priority)

    def update_priorities(self, priorities, saved_idxes):
        assert not np.any(np.isnan(priorities)), priorities
        with self._locker:
            if self._to_update_top_priority:
                self._top_priority = max(self.top_priority, np.max(priorities))
            self._data_structure.batch_update(saved_idxes, priorities)

    """ Implementation """
    def _update_beta(self):
        self._beta = self._beta_schedule.value(self.sample_i)

    @override(Replay)
    def _merge(self, local_buffer, length):
        assert np.all(local_buffer['priority'][: length] != 0)
        # update sum tree
        mem_idxes = np.arange(self.mem_idx, self._mem_idx + length) % self._capacity
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
        self._data_structure = SumTree(self.capacity)        # mem_idx    -->     priority

    """ Implementation """
    @override(PERBase)
    def _sample(self, batch_size=None):
        batch_size = batch_size or self._batch_size
        total_priorities = self._data_structure.total_priorities

        intervals = np.linspace(0, total_priorities, batch_size+1)
        values = np.random.uniform(intervals[:-1], intervals[1:])
        priorities, indexes = self._data_structure.batch_find(values)

        probabilities = priorities / total_priorities

        # compute importance sampling ratios
        IS_ratios = self._compute_IS_ratios(probabilities)
        samples = self._get_samples(indexes)
        samples['IS_ratio'] = IS_ratios
        samples['saved_idxes'] = indexes
        
        return samples
