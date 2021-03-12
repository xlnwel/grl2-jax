import numpy as np

from utility.utils import moments, standardize
from algo.ppo.buffer import Buffer as PPOBuffer


def compute_nae(reward, discount, value, last_value, traj_ret, gamma):
    next_return = last_value
    for i in reversed(range(reward.shape[1])):
        traj_ret[:, i] = next_return = (reward[:, i]
            + discount[:, i] * gamma * next_return)

    # Standardize traj_ret and advantages
    traj_ret_mean, traj_ret_var = moments(traj_ret)
    traj_ret_std = np.maximum(np.sqrt(traj_ret_var), 1e-8)
    value = standardize(value)
    # To have the same mean and std as trajectory return
    value = (value + traj_ret_mean) / traj_ret_std
    advantage = standardize(traj_ret - value)
    traj_ret = standardize(traj_ret)
    return traj_ret, advantage

def compute_gae(reward, discount, value, last_value, gamma, gae_discount, kl):
    next_value = np.concatenate(
            [value[:, 1:], np.expand_dims(last_value, 1)], axis=1)
    advs = delta = (reward - kl + discount * gamma * next_value - value)
    next_adv = 0
    for i in reversed(range(advs.shape[1])):
        advs[:, i] = next_adv = (delta[:, i] 
            + discount[:, i] * gae_discount * next_adv)
    traj_ret = advs + value
    advs += kl
    advantage = standardize(advs)
    return traj_ret, advantage

def compute_indices(idxes, mb_idx, mb_size, N_MBS):
    start = mb_idx * mb_size
    end = (mb_idx + 1) * mb_size
    mb_idx = (mb_idx + 1) % N_MBS
    curr_idxes = idxes[start: end]
    return mb_idx, curr_idxes


class Buffer(PPOBuffer):
    def finish(self, last_value):
        assert self._idx == self.N_STEPS, self._idx
        self.reshape_to_store()
        kl = self._kl_coef * self._memory['kl']
        if self._adv_type == 'nae':
            self._memory['advantage'], self._memory['traj_ret'] = \
                compute_nae(reward=self._memory['reward'], 
                            discount=self._memory['discount'],
                            value=self._memory['value'],
                            last_value=last_value,
                            traj_ret=self._memory['traj_ret'],
                            gamma=self._gamma)
        elif self._adv_type == 'gae':
            self._memory['traj_ret'], self._memory['advantage'] = \
                compute_gae(reward=self._memory['reward'], 
                            discount=self._memory['discount'],
                            value=self._memory['value'],
                            last_value=last_value,
                            gamma=self._gamma,
                            gae_discount=self._gae_discount,
                            kl=kl)
        else:
            raise NotImplementedError

        self.reshape_to_sample()
        self._ready = True
