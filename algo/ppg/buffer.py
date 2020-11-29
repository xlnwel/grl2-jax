from algo.ppo.buffer import *


class Storage:
    def __init__(self, config, **kwargs):
        self.N_PI = config['N_PI']
        self._n_segs = config['n_segs']
        buff_size = config['n_envs'] * config['N_STEPS']
        self._size = buff_size * self._n_segs
        self._n_mbs = config['N_AUX_MBS']
        self._mb_size = self._size // self._n_mbs
        self._idxes = np.arange(self._size)
        self._shuffled_idxes = np.arange(self._size)
        self._idx = 0
        self._mb_idx = 0
        self._ready = False

        self._gamma = config['gamma']
        self._gae_discount = config['gamma'] * config['lam']
        self._buff = Buffer(config, **kwargs)
        self._memory = {}

    def ready(self):
        return self._ready

    def add(self, **data):
        self._buff.add(**data)

    def update(self, key, value, field='mb', mb_idx=None):
        self._buff.update(key, value, field, mb_idx)
    
    def update_value_with_func(self, fn):
        self.buff.update_value_with_func(fn)
    
    def compute_aux_data_with_func(self, fn):
        value_list = []
        logits_list = []
        for start in range(0, self._size, self._mb_size):
            end = start + self._mb_size
            idxes = self._idxes[start:end]
            obs = self._memory['obs'][idxes]
            logits, value = fn(obs)
            value_list.append(value)
            logits_list.append(logits)
        value = np.concatenate(value_list, axis=0)
        self.store('value', value)
        logits = np.concatenate(logits_list, axis=0)
        self.store('logits', logits)

    def store(self, key, value=None):
        if value is None:
            assert self._buff.ready()
        value = value or self._buff[key]
        if key in self._memory:
            self._memory[key] = np.concatenate(
                [self._memory[key], value], axis=0
            )
        else:
            self._memory[key] = value.copy()
        assert self._memory[key].shape[0] <= self._size, \
            (self._memory[key].shape, self._idx, self._size)
    
    def transfer_data(self):
        assert self._buff.ready()
        # do not move the following code to finish function
        # as finish function may be called multiple times to recompute values
        if self._idx >= self.N_PI - self._n_segs:
            for k in ('obs', 'reward', 'discount'):
                self.store(k)
        self._idx = (self._idx + 1) % self.N_PI
        if self._idx == 0:
            self._ready = True

    def sample(self):
        return self._buff.sample()
    
    def sample_aux_data(self):
        assert self._ready, self._idx
        if self._mb_idx == 0:
            np.random.shuffle(self._shuffled_idxes)
        self._mb_idx, self._curr_idxes = compute_indices(
            self._shuffled_idxes, self._mb_idx, self._mb_size, self._n_mbs)
        return {k: self._memory[k][self._curr_idxes] for k in
            ('obs', 'logits', 'traj_ret')}

    def finish(self, last_value):
        self._buff.finish(last_value)
    
    def aux_finish(self, last_value):
        assert self._ready, self._idx
        self._memory['traj_ret'], _ = \
            compute_gae(reward=self._memory['reward'], 
                        discount=self._memory['discount'],
                        value=self._memory['value'],
                        last_value=last_value,
                        gamma=self._gamma,
                        gae_discount=self._gae_discount)

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
    store = Storage(config)
    i = 0
    while not store.ready():
        i += 1
        print(i, store._idx)
        store.reset()
        for _ in range(n_steps):
            store.add(
                obs=np.ones((2, 64, 64, 3)) * i, 
                reward=np.random.uniform(size=(2,)),
                value=np.random.uniform(size=(2,)),
                discount=np.random.uniform(size=(2,)))
        store.finish(np.random.uniform(size=(2,)))
        
    print('obs' in store._memory)
    if 'obs' in store._memory:
        print(store._memory['obs'].shape)
    for i in range(16):
        print(store._memory['obs'][i*100, 0, 0, 0])