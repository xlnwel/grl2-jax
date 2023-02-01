from functools import partial
import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
import haiku as hk

from core.log import do_logging
from core.elements.trainer import create_trainer
from core import optimizer
from core.typing import AttrDict, dict2AttrDict
from tools.display import print_dict_info
from tools.timer import Timer
from tools.utils import flatten_dict
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
    data.action = jnp.zeros((*basic_shape, action_dim), jnp.float32)
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
    def lookahead_train(self, data: AttrDict):
        # NOTE: we utilze the params
        theta = self.model.params.copy()
        is_lookahead = theta.pop('lookahead')
        assert is_lookahead == False, is_lookahead
        opt_state = self.params.theta
        for _ in range(self.config.n_lookahead_epochs):
            np.random.shuffle(self.indices)
            indices = np.split(self.indices, self.config.n_mbs)
            for idx in indices:
                with Timer('lookahead_train'):
                    d = data.slice(idx)
                    if self.config.popart:
                        d.popart_mean = self.popart.mean
                        d.popart_std = self.popart.std
                    theta, opt_state, _ = \
                        self.jit_img_train(
                            theta, 
                            opt_state=opt_state,
                            data=d, 
                        )
        
        # NOTE: the updated parameters are valued to lookahead parameters
        for k, v in theta.items():
            self.model.lookahead_params[k] = v


create_trainer = partial(create_trainer,
    name='lka_v2', trainer_cls=Trainer
)


if __name__ == '__main__':
    import haiku as hk
    from tools.yaml_op import load_config
    from env.func import create_env
    from .model import create_model
    from .loss import create_loss
    from core.log import pwc
    config = load_config('algo/ppo/configs/magw_a2c')
    config = load_config('distributed/sync/configs/smac')
    
    env = create_env(config.env)
    model = create_model(config.model, env.stats())
    loss = create_loss(config.loss, model)
    trainer = create_trainer(config.trainer, env.stats(), loss)
    data = construct_fake_data(env.stats(), 0)
    rng = jax.random.PRNGKey(0)
    pwc(hk.experimental.tabulate(trainer.jit_train)(
        model.theta, rng, trainer.params.theta, data), color='yellow')
    # data = construct_fake_data(env.stats(), 0, True)
    # pwc(hk.experimental.tabulate(trainer.raw_meta_train)(
    #     model.eta, model.theta, trainer.params, data), color='yellow')
