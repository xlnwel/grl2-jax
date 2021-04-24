import logging
import numpy as np

from algo.ppo2.buffer import Buffer as BufferBase


logger = logging.getLogger(__name__)

class Buffer(BufferBase):
    def _add_attributes(self):
        self._sample_size = getattr(self, '_sample_size', None)
        self._n_real_envs = self._n_envs    # n_envs takes into account n_agents, i.e., n_envs = n_real_envs * n_agents
        if self._sample_size:
            assert self._n_envs * self.N_STEPS % self._sample_size == 0, \
                f'{self._n_envs} * {self.N_STEPS} % {self._sample_size} != 0'
            size = self._n_envs * self.N_STEPS // self._sample_size
            logger.info(f'Sample size: {self._sample_size}')
        else:
            size = self._n_envs * self.N_STEPS
        self._size = size
        self._mb_size = size // self.N_MBS
        self._gae_discount = self._gamma * self._lam
        self._memory = {}
        self._is_store_shape = True
        self._inferred_sample_keys = False
        self.reset()
        logger.info(f'Batch size(without taking into account #agents): {size}')
        logger.info(f'Mini-batch size(without taking into account #agents): {self._mb_size}')

    def update_value_with_func(self, fn):
        assert self._mb_idx == 0, f'Unfinished sample: self._mb_idx({self._mb_idx}) != 0'
        mb_idx = 0
        for start in range(0, self._size, self._mb_size):
            end = start + self._mb_size
            curr_idxes = self._idxes[start:end]
            shared_state = self._memory['shared_state'][curr_idxes]
            state = (self._memory[k][curr_idxes, 0] for k in self._state_keys)
            mask = self._memory['mask'][curr_idxes]
            value, state = fn(shared_state, state=state, mask=mask)
            self.update('value', value, mb_idxes=curr_idxes)
            # NOTE: you may want to update states as well
        
        assert mb_idx == 0, mb_idx

    def finish(self, last_value):
        assert self._idx == self.N_STEPS, self._idx
        self.reshape_to_store()
        self._memory['advantage'], self._memory['traj_ret'] = \
            self._compute_advantage_return(
                self._memory['reward'], self._memory['discount'], 
                self._memory['value'], last_value, 
                mask=self._memory['life_mask']
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
