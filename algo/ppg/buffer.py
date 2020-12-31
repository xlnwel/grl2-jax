from algo.ppo.buffer import *


class Replay:
    def __init__(self, config, **kwargs):
        self.N_PI = config['N_PI']
        self._n_segs = config['n_segs']
        self._n_envs = config['n_envs']
        self.N_STEPS = config['N_STEPS'] * self._n_segs
        buff_size = config['n_envs'] * config['N_STEPS']
        self._size = buff_size * self._n_segs
        self._n_mbs = self._n_segs * config['N_AUX_MBS_PER_SEG']
        self._mb_size = self._size // self._n_mbs
        self._shuffled_idxes = np.arange(self._size)
        self._idx = 0
        self._mb_idx = 0
        self._ready = False

        self._gamma = config['gamma']
        self._gae_discount = config['gamma'] * config['lam']
        self._buff = Buffer(config, **kwargs)
        self._memory = {}

    def __getitem__(self, k):
        return self._buff[k]

    def ready(self):
        return self._ready

    def add(self, **data):
        self._buff.add(**data)

    def update(self, key, value, field='mb', mb_idx=None):
        self._buff.update(key, value, field, mb_idx)
    
    def aux_update(self, key, value, field='mb', mb_idxes=None):
        if field == 'mb':
            mb_idxes = self._curr_idxes if mb_idxes is None else mb_idxes
            self._memory[key][mb_idxes] = value
        elif field == 'all':
            assert self._memory[key].shape == value.shape, (self._memory[key].shape, value.shape)
            self._memory[key] == value
        else:
            raise ValueError(f'Unknown field: {field}. Valid fields: ("all", "mb")')

    def update_value_with_func(self, fn):
        self.buff.update_value_with_func(fn)
    
    def compute_aux_data_with_func(self, fn):
        assert self._idx == 0, self._idx
        value_list = []
        logits_list = []
        self._memory = flatten_time_dim(self._memory, self._n_envs, self.N_STEPS)
        for start in range(0, self._size, self._mb_size):
            end = start + self._mb_size
            obs = self._memory['obs'][start:end]
            logits, value = fn(obs)
            value_list.append(value)
            logits_list.append(logits)
        self._memory['value'] = np.concatenate(value_list, axis=0)
        self._memory['logits'] = np.concatenate(logits_list, axis=0)
    
    def transfer_data(self):
        assert self._buff.ready()
        self._buff.restore_time_dim()
        if self._idx >= self.N_PI - self._n_segs:
            for k in ('obs', 'reward', 'discount'):
                v = self._buff[k]
                if k in self._memory:
                    self._memory[k] = np.concatenate(
                        [self._memory[k], v], axis=1
                    )
                else:
                    self._memory[k] = v.copy()
        self._idx = (self._idx + 1) % self.N_PI

    def sample(self):
        return self._buff.sample()
    
    def sample_aux_data(self):
        assert self._ready, self._idx
        if self._mb_idx == 0:
            np.random.shuffle(self._shuffled_idxes)
        self._mb_idx, self._curr_idxes = compute_indices(
            self._shuffled_idxes, self._mb_idx, self._mb_size, self._n_mbs)
        return {k: self._memory[k][self._curr_idxes] for k in
            ('obs', 'logits', 'value', 'traj_ret')}

    def finish(self, last_value):
        self._buff.finish(last_value)
    
    def aux_finish(self, last_value):
        assert self._idx == 0, self._idx
        self._memory = restore_time_dim(self._memory, self._n_envs, self.N_STEPS)
        self._memory['traj_ret'], _ = \
            compute_gae(reward=self._memory['reward'], 
                        discount=self._memory['discount'],
                        value=self._memory['value'],
                        last_value=last_value,
                        gamma=self._gamma,
                        gae_discount=self._gae_discount)
        self._memory = flatten_time_dim(self._memory, self._n_envs, self.N_STEPS)
        for k, v in self._memory.items():
            assert v.shape[0] == self._n_envs * self.N_STEPS, (k, v.shape)
        self._ready = True

    def reset(self):
        if self._buff.ready():
            self.transfer_data()
        self._buff.reset()
    
    def aux_reset(self):
        assert self._ready, self._idx
        self._ready = False
        self._idx = 0
        self._memory.clear()
    
    def clear(self):
        self._idx = 0
        self._ready = False
        self._buff.clear()
        self._memory.clear()


if __name__ == '__main__':
    n_envs = 2
    n_steps = 50
    n_mbs = 4
    config = dict(
        N_PI=16,
        N_AUX_MBS=16,
        n_segs=16,
        adv_type='gae',
        gamma=.99,
        lam=.95,
        n_envs=n_envs,
        N_STEPS=n_steps,
        N_MBS=n_mbs,
    )
    replay = Replay(config)
    i = 0
    while not replay.ready():
        i += 1
        print(i, replay._idx)
        replay.reset()
        for _ in range(n_steps):
            replay.add(
                obs=np.ones((2, 64, 64, 3)) * i, 
                reward=np.random.uniform(size=(2,)),
                value=np.random.uniform(size=(2,)),
                discount=np.random.uniform(size=(2,)))
        replay.finish(np.random.uniform(size=(2,)))
        
    print('obs' in replay._memory)
    if 'obs' in replay._memory:
        print(replay._memory['obs'].shape)
    for i in range(16):
        print(replay._memory['obs'][i*100, 0, 0, 0])