import numpy as np

class LocalBuffer:
    def __init__(self, 
                n_envs, 
                seqlen,
                burn_in,
                state_shape,
                state_dtype,
                action_shape,
                action_dtype):
        self.n_envs = n_envs
        self.seqlen = seqlen
        self.burn_in = burn_in
        
        basic_shape = (n_envs, seqlen)
        self.memory = dict(
            state=np.ndarray((n_envs, seqlen), dtype=object),
            action=np.ndarray((n_envs, seqlen), dtype=object),
            reward=np.ndarray((n_envs, seqlen), dtype=np.float32),
            done=np.zeros((n_envs, seqlen), dtype=np.bool),
            steps=np.zeros((n_envs, seqlen), dtype=np.uint8),
            next_state=np.ndarray((n_envs, seqlen), dtype=object),
            mask=np.zeros((n_envs, seqlen), dtype=np.bool)
        )

        self.indices = np.zeros(n_envs, dtype=np.int32) + burn_in

    def add(self, **data):
        for k, v in data.items():
            self.memory[k][:, self.indices] = v
        self.memory['mask'][:, self.indices] = 1
        
        self.indices += 1

        m = np.where(self.indices == self.seqlen, 1, 0)
        state = self.memory['state'][m]

    def reset(self, env_id):
        self.indices[env_id] = 0
        self.memory['mask'][env_id] = np.zeros(self.seqlen, dtype=np.float32)
