import numpy as np

from utility.decorators import override
from replay.basic_replay import Replay
from replay.utils import init_buffer


class UniformReplay(Replay):
    """ Interface """
    def __init__(self, config, state_shape, state_dtype, 
                action_dim, action_dtype, gamma, 
                has_next_state=False):
        super().__init__(config, state_shape, action_dim, gamma)

        init_buffer(self.memory, self.capacity, state_shape, state_dtype, 
                    action_dim, action_dtype, False, 
                    has_next_state=has_next_state)

        # Code for single agent
        if self.n_steps > 1:
            self.tb_capacity = config['tb_capacity']
            self.tb_idx = 0
            self.tb_full = False
            self.tb = {}
            init_buffer(self.tb, self.tb_capacity, state_shape, state_dtype, 
                        action_dim, action_dtype, False, 
                        has_next_state=has_next_state)

    @override(Replay)
    def add(self, state, action, reward, done, next_state=None):
        super()._add(state, action, reward, done, next_state=next_state)

    """ Implementation """
    @override(Replay)
    def _sample(self):
        size = self.capacity if self.is_full else self.mem_idx
        indexes = np.random.randint(0, size, self.batch_size)
        
        samples = self._get_samples(indexes)

        return samples
