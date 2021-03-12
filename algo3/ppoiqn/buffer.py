import logging
import numpy as np

from algo.ppo.buffer import Buffer as PPOBuffer, compute_nae, compute_gae

logger = logging.getLogger(__name__)


class Buffer(PPOBuffer):
    def finish(self, last_value):
        assert self._idx == self.N_STEPS, self._idx
        self.reshape_to_store()
        reward = np.expand_dims(self._memory['reward'], -1)
        discount = np.expand_dims(self._memory['discount'], -1)
        if self._adv_type == 'nae':
            self._memory['advantage'], self._memory['traj_ret'] = \
                compute_nae(reward=reward, 
                            discount=discount,
                            value=self._memory['value'],
                            last_value=last_value,
                            traj_ret=self._memory['traj_ret'],
                            gamma=self._gamma)
        elif self._adv_type == 'gae':
            self._memory['traj_ret'], self._memory['advantage'] = \
                compute_gae(reward=reward, 
                            discount=discount,
                            value=self._memory['value'],
                            last_value=last_value,
                            gamma=self._gamma,
                            gae_discount=self._gae_discount)
        else:
            raise NotImplementedError

        self.reshape_to_sample()
        self._ready = True