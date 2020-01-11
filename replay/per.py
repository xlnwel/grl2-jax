from threading import Lock
import numpy as np

from utility.decorators import override
from utility.schedule import PiecewiseSchedule
from replay.base import Replay
from replay.ds.sum_tree import SumTree


class PERBase(Replay):
    """ Base class for PER, left in case one day I implement rank-based PER """
    def __init__(self, config, *keys, state_shape=None):
        super().__init__(config, *keys, state_shape=state_shape)
        self.data_structure = None            

        # params for prioritized replay
        self.beta = float(config.get('beta0', .4))
        self.beta_schedule = PiecewiseSchedule([(0, self.beta), (float(config['beta_steps']), 1.)], 
                                                outside_value=1.)

        self.top_priority = 2.
        self.to_update_top_priority = config['to_update_top_priority'] if 'to_update_top_priority' in config else True

        self.sample_i = 0   # count how many times self.sample is called

        # locker used to avoid conflict introduced by tf.data.Dataset
        # used to ensure SumTree update will not happen when sampling
        # which may cause out-of-range sampling when calling data_structure.find
        self.locker = Lock()

    @override(Replay)
    def sample(self):
        assert self.good_to_learn(), (
            'There are not sufficient transitions to start learning --- '
            f'transitions in buffer({len(self)}) vs '
            f'minimum required size({self.min_size})')
        with self.locker:
            samples = self._sample()
            self.sample_i += 1
            self._update_beta()

        return samples

    @override(Replay)
    def add(self, **kwargs):
        # it is okay to add when sampling, so no locker is needed
        super().add(**kwargs)
        # super().add updates self.mem_idx 
        self.data_structure.update(self.top_priority, self.mem_idx - 1)

    def update_priorities(self, priorities, saved_indices):
        assert not np.any(np.isnan(priorities)), priorities
        with self.locker:
            if self.to_update_top_priority:
                self.top_priority = max(self.top_priority, np.max(priorities))
            for priority, idx in zip(priorities, saved_indices):
                self.data_structure.update(priority, idx)

    """ Implementation """
    def _update_beta(self):
        self.beta = self.beta_schedule.value(self.sample_i)

    @override(Replay)
    def _merge(self, local_buffer, length):
        end_idx = self.mem_idx + length
        assert np.all(local_buffer['priority'][: length] != 0)
        for idx, mem_idx in enumerate(range(self.mem_idx, end_idx)):
            self.data_structure.update(local_buffer['priority'][idx], mem_idx % self.capacity)
            
        super()._merge(local_buffer, length)
        
    def _compute_IS_ratios(self, probabilities):
        IS_ratios = (np.min(probabilities) / probabilities)**self.beta

        return IS_ratios


class ProportionalPER(PERBase):
    """ Interface """
    def __init__(self, config, *keys, state_shape=None):
        super().__init__(config, *keys, state_shape=state_shape)
        self.data_structure = SumTree(self.capacity)        # mem_idx    -->     priority

    """ Implementation """
    @override(PERBase)
    def _sample(self):
        total_priorities = self.data_structure.total_priorities
        
        segment = total_priorities / self.batch_size

        # priorities, indexes = [], []
        # vs = []
        # for k in range(self.batch_size):
        #     v = np.random.uniform(k * segment, (k+1) * segment)
        #     vs.append(v)
        #     p, i = self.data_structure.find(v)
        #     priorities.append(p)
        #     indexes.append(i)
        #     if i > self.mem_idx or p == 0:
        #         print('k', k)
        #         print('segment', segment, k * segment, (k+1) * segment)
        #         print('v', v)
        #         print('priority', p)
        #         print('i', i, self.mem_idx)
        #         import sys
        #         sys.exit()

        priorities, indexes = list(zip(
            *[self.data_structure.find(np.random.uniform(i * segment, (i+1) * segment))
                for i in range(self.batch_size)]))

        np.testing.assert_array_less(np.zeros_like(priorities), priorities)

        priorities = np.array(priorities)
        probabilities = priorities / total_priorities

        # compute importance sampling ratios
        IS_ratios = self._compute_IS_ratios(probabilities)
        samples = self._get_samples(indexes)
        
        return IS_ratios, indexes, samples
