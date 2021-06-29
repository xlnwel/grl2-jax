import logging
import numpy as np

from algo.ppo.buffer import Buffer as BufferBase, \
    compute_indices, standardize


logger = logging.getLogger(__name__)

class Buffer(BufferBase):
    def update_value_with_func(self, fn):
        assert self._mb_idx == 0, f'Unfinished sample: self._mb_idx({self._mb_idx}) != 0'
        mb_idx = 0
        for start in range(0, self._size, self._mb_size):
            end = start + self._mb_size
            curr_idxes = self._idxes[start:end]
            global_state = self._memory['global_state'][curr_idxes]
            state = tuple([self._memory[k][curr_idxes, 0] for k in self._state_keys])
            mask = self._memory['mask'][curr_idxes]
            value = fn(global_state, state=state, mask=mask)
            self.update('value', value, mb_idxes=curr_idxes)
            # NOTE: you may want to update states as well
        
        assert mb_idx == 0, mb_idx

    def sample(self, sample_keys=None):
        assert self._ready
        if self._mb_idx == 0:
            np.random.shuffle(self._shuffled_idxes)

        sample_keys = sample_keys or self._sample_keys
        self._mb_idx, self._curr_idxes = compute_indices(
            self._shuffled_idxes, self._mb_idx, 
            self._mb_size, self.N_MBS)
        
        sample = {k: self._memory[k][self._curr_idxes, 0]
            if k in self._state_keys 
            else self._memory[k][self._curr_idxes] 
            for k in sample_keys}
        
        if 'advantage' in sample and self._norm_adv == 'minibatch':
            sample['advantage'] = standardize(
                sample['advantage'], mask=sample['life_mask'], epsilon=self._epsilon)
        
        return sample

    def finish(self, last_value):
        assert self._idx == self.N_STEPS, self._idx
        self.reshape_to_store()
        self._memory['advantage'], self._memory['traj_ret'] = \
            self._compute_advantage_return(
                self._memory['reward'], self._memory['discount'], 
                self._memory['value'], last_value, 
                mask=self._memory.get('life_mask'),
                epsilon=self._epsilon
            )

        self.reshape_to_sample()
        self._ready = True

    def _init_buffer(self, data):
        self._n_envs = data['reward'].shape[0]
        self._size = self._n_envs * self.N_STEPS // self._sample_size
        self._mb_size = self._size // self.N_MBS
        self._idxes = np.arange(self._size)
        self._shuffled_idxes = np.arange(self._size)

        super()._init_buffer(data)
