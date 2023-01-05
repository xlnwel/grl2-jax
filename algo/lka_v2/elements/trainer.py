from functools import partial
import numpy as np
import jax.numpy as jnp

from core.elements.trainer import create_trainer
from core.typing import AttrDict, dict2AttrDict
from tools.display import print_dict_info
from tools.timer import Timer
from algo.zero.elements.trainer import Trainer as TrainerBase


def construct_fake_data(env_stats, aid):
    b = 8
    s = 400
    u = len(env_stats.aid2uids[aid])
    shapes = env_stats.obs_shape[aid]
    dtypes = env_stats.obs_dtype[aid]
    action_dim = env_stats.action_dim[aid]
    basic_shape = (b, s, u)
    data = {k: jnp.zeros((b, s+1, u, *v), dtypes[k]) 
        for k, v in shapes.items()}
    data = dict2AttrDict(data)
    data.setdefault('global_state', data.obs)
    data.action = jnp.zeros(basic_shape, jnp.int32)
    data.value = jnp.zeros(basic_shape, jnp.float32)
    data.reward = jnp.zeros(basic_shape, jnp.float32)
    data.discount = jnp.zeros(basic_shape, jnp.float32)
    data.reset = jnp.zeros(basic_shape, jnp.float32)
    data.mu_logprob = jnp.zeros(basic_shape, jnp.float32)
    data.mu_logits = jnp.zeros((*basic_shape, action_dim), jnp.float32)
    data.advantage = jnp.zeros(basic_shape, jnp.float32)
    data.v_target = jnp.zeros(basic_shape, jnp.float32)

    print_dict_info(data)
    
    return data


class Trainer(TrainerBase):
    def imaginary_train(self, data: AttrDict):
        # NOTE: we utilze the params
        theta = self.model.params.copy()
        is_imaginary = theta.pop('imaginary')
        assert is_imaginary == False, is_imaginary
        opt_state = self.imaginary_opt_state
        if self.config.popart:
            data.popart_mean = self.popart.mean
            data.popart_std = self.popart.std
        for _ in range(self.config.n_imaginary_epochs):
            np.random.shuffle(self.indices)
            indices = np.split(self.indices, self.config.n_mbs)
            for idx in indices:
                d = data.slice(idx)
                if self.config.popart:
                    d.popart_mean = self.popart.mean
                    d.popart_std = self.popart.std
                with Timer('imaginary_train'):
                    theta, opt_state, _ = \
                        self.jit_img_train(
                            theta, 
                            opt_state=opt_state,
                            data=d, 
                        )
        
        # NOTE: the updated parameters are valued to imaginary parameters
        for k, v in theta.items():
            self.model.imaginary_params[k] = v
        self.imaginary_opt_state = opt_state


create_trainer = partial(create_trainer,
    name='lka_v2', trainer_cls=Trainer
)
