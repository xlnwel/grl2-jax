import numpy as np

from utility.decorators import override
from utility.display import assert_colorize
from utility.schedule import PiecewiseSchedule
from replay.basic_replay import Replay
from replay.utils import init_buffer, add_buffer, copy_buffer
from replay.ds.sum_tree import SumTree


class PERBase(Replay):
    """ Base class for PER, left in case one day I implement rank-based PER """
    def __init__(self, config, state_shape, state_dtype, 
                action_dim, action_dtype, gamma, 
                has_next_state=False):
        super().__init__(config, state_shape, action_dim, gamma)
        self.data_structure = None            

        # params for prioritized replay
        self.beta = float(config['beta0']) if 'beta0' in config else .4
        self.beta_schedule = PiecewiseSchedule([(0, config['beta0']), (float(config['beta_steps']), 1.)], 
                                                outside_value=1.)

        self.top_priority = 2.
        self.to_update_priority = config['to_update_priority'] if 'to_update_priority' in config else True

        self.sample_i = 0   # count how many times self.sample is called

        init_buffer(self.memory, self.capacity, state_shape, state_dtype, 
                    action_dim, action_dtype, self.n_steps == 1, 
                    has_next_state=has_next_state)

        # Code for single agent
        if 'tb_capacity' in config and self.n_steps > 1:
            self.tb_capacity = config['tb_capacity']
            self.tb_idx = 0
            self.tb_full = False
            self.tb = {}
            init_buffer(self.tb, self.tb_capacity, state_shape, state_dtype, 
                        action_dim, action_dtype, True, 
                        has_next_state=has_next_state)

    @override(Replay)
    def sample(self):
        assert_colorize(self.good_to_learn, 'There are not sufficient transitions to start learning --- '
                                            f'transitions in buffer: {len(self)}\t'
                                            f'minimum required size: {self.min_size}')
        with self.locker:        
            samples = self._sample()
            self.sample_i += 1
            self._update_beta()

        return samples

    @override(Replay)
    def add(self, state, action, reward, done, next_state=None):
        if self.n_steps > 1:
            assert_colorize(hasattr(self, 'tb'), 'please specify tb_capacity in config.yaml')
            self.tb['priority'][self.tb_idx] = self.top_priority
        else:
            self.memory['priority'][self.mem_idx] = self.top_priority
            self.data_structure.update(self.top_priority, self.mem_idx)
        super()._add(state, action, reward, done, next_state=next_state)

    def update_priorities(self, priorities, saved_indices):
        with self.locker:
            if self.to_update_priority:
                self.top_priority = max(self.top_priority, np.max(priorities))
            for priority, idx in zip(priorities, saved_indices):
                self.data_structure.update(priority, idx)

    """ Implementation """
    def _update_beta(self):
        self.beta = self.beta_schedule.value(self.sample_i)

    @override(Replay)
    def _merge(self, local_buffer, length):
        end_idx = self.mem_idx + length
        assert np.all(local_buffer['priority'][: length])
        for idx, mem_idx in enumerate(range(self.mem_idx, end_idx)):
            self.data_structure.update(local_buffer['priority'][idx], mem_idx % self.capacity)
            
        super()._merge(local_buffer, length)
        
    def _compute_IS_ratios(self, probabilities):
        IS_ratios = (np.min(probabilities) / probabilities)**self.beta

        return IS_ratios


class ProportionalPER(PERBase):
    """ Interface """
    def __init__(self, config, state_shape, state_dtype, 
                action_dim, action_dtype, gamma,
                has_next_state=False):
        super().__init__(config, state_shape, state_dtype, 
                        action_dim, action_dtype, gamma, 
                        has_next_state=has_next_state)
        self.data_structure = SumTree(self.capacity)        # mem_idx    -->     priority

    """ Implementation """
    @override(PERBase)
    def _sample(self):
        total_priorities = self.data_structure.total_priorities
        
        segment = total_priorities / self.batch_size

        priorities, indexes = list(zip(*[self.data_structure.find(np.random.uniform(i * segment, (i+1) * segment))
                                        for i in range(self.batch_size)]))

        priorities = np.array(priorities)
        probabilities = priorities / total_priorities

        # compute importance sampling ratios
        IS_ratios = self._compute_IS_ratios(probabilities)
        samples = self._get_samples(indexes)
        
        return IS_ratios, indexes, samples
