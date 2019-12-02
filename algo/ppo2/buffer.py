import numpy as np
from copy import deepcopy

from utility.display import assert_colorize, pwc
from utility.utils import moments, standardize


class PPOBuffer(dict):
    def __init__(self, 
                config,
                n_envs, 
                seqlen, 
                state_shape, 
                state_dtype, 
                action_shape, 
                action_dtype):
        self.n_envs = n_envs
        self.seqlen = seqlen

        self.advantage_type = config['advantage_type']
        self.gamma = config['gamma']
        self.gae_discount = self.gamma * config['lam'] 
        # Environment hack
        self.reward_scale = config.get('reward_scale', 1)
        self.reward_clip = config.get('reward_clip')

        self.basic_shape = (n_envs, seqlen)
        self.memory = dict(
            state=np.zeros((*self.basic_shape, *state_shape), dtype=state_dtype),
            action=np.zeros((*self.basic_shape, *action_shape), dtype=action_dtype),
            reward=np.zeros((*self.basic_shape, 1), dtype=np.float32),
            nonterminal=np.zeros((*self.basic_shape, 1), dtype=np.float32),
            value=np.zeros((n_envs, seqlen+1, 1), dtype=np.float32),
            traj_ret=np.zeros((*self.basic_shape, 1), dtype=np.float32),
            advantage=np.zeros((*self.basic_shape, 1), dtype=np.float32),
            old_logpi=np.zeros((*self.basic_shape, 1), dtype=np.float32),
            mask=np.zeros((*self.basic_shape, 1), dtype=np.float32),
        )

        self.reset()

    def add(self, **data):
        assert_colorize(self.idx < self.seqlen, 
            f'Out-of-range idx {self.idx}. Call "self.reset" beforehand')
        idx = self.idx

        for k, v in data.items():
            if v is not None:
                self.memory[k][:, idx] = v

        self.idx += 1

    def sample(self):
        assert_colorize(self.ready, 
            f'PPOBuffer is not ready to be read. Call "self.finish" first')

        keys = ['state', 'action', 'traj_ret', 'value', 
                'advantage', 'old_logpi', 'mask']

        return {k: self.memory[k][:, :self.idx]
                for k in keys if self.memory[k] is not None}

    def finish(self, last_value):
        self.memory['value'][:, self.idx] = last_value
        valid_slice = np.s_[:, :self.idx]
        self.memory['mask'][:, self.idx:] = 0
        mask = self.memory['mask'][valid_slice]

        # Environment hack
        self.memory['reward'] *= self.reward_scale
        if self.reward_clip:
            self.memory['reward'] = np.clip(self.memory['reward'], -self.reward_clip, self.reward_clip)

        if self.advantage_type == 'nae':
            traj_ret = self.memory['traj_ret'][valid_slice]
            next_return = last_value
            for i in reversed(range(self.idx)):
                traj_ret[:, i] = next_return = (self.memory['reward'][:, i] 
                    + self.memory['nonterminal'][:, i] * self.gamma * next_return)

            # Standardize traj_ret and advantages
            traj_ret_mean, traj_ret_std = moments(traj_ret, mask=mask)
            value = standardize(self.memory['value'][valid_slice], mask=mask)
            # To have the same mean and std as trajectory return
            value = (value + traj_ret_mean) / (traj_ret_std + 1e-8)     
            self.memory['advantage'][valid_slice] = standardize(traj_ret - value, mask=mask)
            self.memory['traj_ret'][valid_slice] = standardize(traj_ret, mask=mask)
        elif self.advantage_type == 'gae':
            advs = delta = (self.memory['reward'][valid_slice] 
                + self.memory['nonterminal'][valid_slice] * self.gamma * self.memory['value'][:, 1:self.idx+1]
                - self.memory['value'][valid_slice])
            next_adv = 0
            for i in reversed(range(self.idx)):
                advs[:, i] = next_adv = delta[:, i] + self.memory['nonterminal'][:, i] * self.gae_discount * next_adv
            self.memory['traj_ret'][valid_slice] = advs + self.memory['value'][valid_slice]
            self.memory['advantage'][valid_slice] = standardize(advs, mask=mask)
            # Code for double check 
            # mb_returns = np.zeros_like(mask)
            # mb_advs = np.zeros_like(mask)
            # lastgaelam = 0
            # for t in reversed(range(self.idx)):
            #     if t == self.idx - 1:
            #         nextnonterminal = self.memory['nonterminal'][:, t]
            #         nextvalues = last_value
            #     else:
            #         nextnonterminal = self.memory['nonterminal'][:, t]
            #         nextvalues = self.memory['value'][:, t+1]
            #     delta = self.memory['reward'][:, t] + self.gamma * nextvalues * nextnonterminal - self.memory['value'][:, t]
            #     mb_advs[:, t] = lastgaelam = delta + self.gae_discount * nextnonterminal * lastgaelam
            # mb_advs = standardize(mb_advs, mask=mask)
            # assert np.all(np.abs(mb_advs - self.memory['advantage'][valid_slice])<1e-4), f'{mb_advs.flatten()}\n{self.memory["advantage"][valid_slice].flatten()}'
        else:
            raise NotImplementedError

        for k, v in self.memory.items():
            v[valid_slice] = (v[valid_slice].T * mask.T).T
        
        self.ready = True

    def reset(self):
        self.idx = 0
        self.ready = False      # Whether the buffer is ready to be read

    def good_to_learn(self):
        return np.sum(self.memory['mask']) > 500


if __name__ == '__main__':
    gamma = .99
    lam = .95
    gae_discount = gamma * lam
    config = dict(
        gamma=gamma,
        lam=lam,
        advantage_type='gae'
    )
    kwargs = dict(
        config=config,
        n_envs=8, 
        seqlen=1000, 
        n_minibatches=2, 
        state_shape=[3], 
        state_dtype=np.float32, 
        action_shape=[2], 
        action_dtype=np.float32,
    )
    buffer = PPOBuffer(**kwargs)
    d = np.zeros((kwargs['n_envs'], 1))
    m = np.ones((kwargs['n_envs'], 1))
    diff = kwargs['seqlen'] - kwargs['n_envs']
    for i in range(kwargs['seqlen']):
        r = np.random.rand(kwargs['n_envs'], 1)
        v = np.random.rand(kwargs['n_envs'], 1)
        if np.random.randint(2):
            d[np.random.randint(kwargs['n_envs'])] = 1
        buffer.add(reward=r,
                value=v,
                nonterminal=1-d,
                mask=m)
        m = 1-d
        if np.all(d == 1):
            break
    last_value = np.random.rand(kwargs['n_envs'], 1)
    buffer.finish(last_value)
    