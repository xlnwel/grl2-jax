import numpy as np

from utility.decorators import override
from utility.display import assert_colorize
from utility.schedule import PiecewiseSchedule
from buffer.replay.basic_replay import Replay
from buffer.replay.utils import init_buffer, add_buffer, copy_buffer


class PrioritizedReplay(Replay):
    """ Interface """
    def __init__(self, config, state_shape, action_dim, gamma):
        super().__init__(config, state_shape, action_dim, gamma)
        self.data_structure = None            

        # params for prioritized replay
        self.beta = float(config['beta0']) if 'beta0' in config else .4
        self.beta_schedule = PiecewiseSchedule([(0, config['beta0']), (float(config['beta_steps']), 1.)], 
                                                outside_value=1.)

        self.top_priority = 2.
        self.to_update_priority = config['to_update_priority'] if 'to_update_priority' in config else True

        self.sample_i = 0   # count how many times self.sample is called

        init_buffer(self.memory, self.capacity, state_shape, action_dim, self.n_steps == 1)

        # Code for single agent
        if self.n_steps > 1:
            self.tb_capacity = config['tb_capacity']
            self.tb_idx = 0
            self.tb_full = False
            self.tb = {}
            init_buffer(self.tb, self.tb_capacity, state_shape, action_dim, True)

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
    def add(self, state, action, reward, done):
        if self.n_steps > 1:
            self.tb['priority'][self.tb_idx] = self.top_priority
        else:
            self.memory['priority'][self.mem_idx] = self.top_priority
            self.data_structure.update(self.top_priority, self.mem_idx)
        super()._add(state, action, reward, done)

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
