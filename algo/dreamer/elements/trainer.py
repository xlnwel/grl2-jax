from functools import partial
import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
import jax.nn as nn
import haiku as hk
from algo.dreamer.elements.model import LOOKAHEAD

from core.log import do_logging, pwc
from core.elements.trainer import TrainerBase, create_trainer
from core import optimizer
from core.typing import dict2AttrDict, AttrDict
from tools.display import print_dict_info
from tools.feature import one_hot
from tools.timer import Timer
from tools.utils import flatten_dict, prefix_name


def construct_fake_data(env_stats, aid):
    b = 8
    s = 400
    u = 2
    shapes = env_stats.obs_shape[aid]
    dtypes = env_stats.obs_dtype[aid]
    basic_shape = (b, s, u)
    data = {k: jnp.zeros((b, s+1, u, *v), dtypes[k]) 
        for k, v in shapes.items()}
    data = dict2AttrDict(data)
    data.setdefault('global_state', data.obs)
    action_dim = env_stats.action_dim[aid]
    data.action = jnp.zeros((*basic_shape, action_dim), jnp.float32)
    data.reset = jnp.zeros(basic_shape, jnp.float32)
    data.reward = jnp.zeros(basic_shape, jnp.float32)

    print_dict_info(data)
    
    return data


class Trainer(TrainerBase):
    def build_optimizers(self):
        theta = self.model.theta.copy()
        self.params.theta = AttrDict()
        theta.policies = [p.copy() for p in theta.policies]
        [p.pop(LOOKAHEAD) for p in theta.policies]
        theta.vs = [v.copy() for v in theta.vs]
        # theta_model = [
        #     theta.rssm_embed, theta.rssm_rnn, theta.rssm_trans, theta.rssm_repre,
        #     theta.reward, theta.discount, theta.state_encoder, theta.obs_encoder, theta.decoder
        # ]
        self.opts.rl, self.params.theta.rl = optimizer.build_optimizer(
            params=theta,
            **self.config.rl_opt,
            name='rl'
        )
        self.opts.model, self.params.theta.model = optimizer.build_optimizer(
            params=theta,
            **self.config.theta_opt,
            name='model'
        )

    def compile_train(self):
        _jit_train = jax.jit(self.theta_train)
        def jit_train(*args, **kwargs):
            self.rng, rng = jax.random.split(self.rng)
            return _jit_train(*args, rng=rng, **kwargs)
        self.jit_train = jit_train
        
        _jit_model_train = jax.jit(self.model_theta_train)
        def jit_model_train(*args, **kwargs):
            self.rng, rng = jax.random.split(self.rng)
            return _jit_model_train(*args, rng=rng, **kwargs) 
        self.jit_model_train = jit_model_train
       
        self.haiku_tabulate()

    def model_train(self, data):
        data = self.process_data(data)
        theta = self.model.theta.copy()
        theta.policies = [p.copy() for p in theta.policies]
        is_lookahead = [p.pop(LOOKAHEAD) for p in theta.policies]
        with Timer(f'{self.name}_model_train'):
            theta, self.params.theta.model, stats = \
                self.jit_model_train(
                    theta, 
                    opt_state=self.params.theta.model,
                    data=data,
                )
        for i, p in enumerate(theta.policies):
            p[LOOKAHEAD] = is_lookahead[i]
        self.model.set_weights(theta)
        return stats

    def train(self, data):
        # data = self.process_data(data)
        theta = self.model.theta.copy()
        theta.policies = [p.copy() for p in theta.policies]
        is_lookahead = [p.pop(LOOKAHEAD) for p in theta.policies]
        with Timer(f'{self.name}_train'):
            theta, self.params.theta.model, stats = \
                self.jit_train(
                    theta, 
                    opt_state=self.params.theta.model,
                    data=data,
                )
        for i, p in enumerate(theta.policies):
            p[LOOKAHEAD] = is_lookahead[i]
        self.model.set_weights(theta)
        return stats

    def model_theta_train(
        self,
        theta,
        rng,
        opt_state,
        data,
    ):
        do_logging('model train is traced', backtrack=4)
        
        theta, opt_state, stats = optimizer.optimize(
            self.loss.model_loss, 
            theta, 
            opt_state, 
            kwargs={
                'rng': rng, 
                'data': data, 
            }, 
            opt=self.opts.model, 
            name='train/theta'
        )
        stats.update({f'data/{k}': lax.stop_gradient(v) 
            for k, v in data.items() if v is not None})

        return theta, opt_state, stats

    def theta_train(
        self, 
        theta, 
        rng, 
        opt_state, 
        data, 
    ):
        do_logging('theta train is traced', backtrack=4)
        theta, opt_state, stats = optimizer.optimize(
            self.loss.rl_loss, 
            theta, 
            opt_state, 
            kwargs={
                'rng': rng, 
                'data': data, 
            }, 
            opt=self.opts.rl, 
            name='train/theta'
        )
        stats.update({f'data/{k}': lax.stop_gradient(v) 
            for k, v in data.items() if v is not None})

        return theta, opt_state, stats

    def process_data(self, data):
        if self.env_stats.is_action_discrete[0]:
            data.action = one_hot(data.action, self.env_stats.action_dim[0])
            
        return data


create_trainer = partial(create_trainer,
    name='model', trainer_cls=Trainer
)


def sample_stats(stats, max_record_size=10):
    # we only sample a small amount of data to reduce the cost
    stats = {k if '/' in k else f'train/{k}': 
        np.random.choice(v.reshape(-1), max_record_size) 
        if isinstance(v, (np.ndarray, jnp.DeviceArray)) else v 
        for k, v in stats.items()}
    return stats


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
