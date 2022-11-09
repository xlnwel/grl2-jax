from functools import partial
import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
import haiku as hk
import chex
import optax

from core.log import do_logging, pwc
from core.elements.trainer import TrainerBase, create_trainer
from core import optimizer
from core.typing import dict2AttrDict
from tools.display import print_dict_info
from tools.timer import Timer


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
    data.mu = jnp.zeros((*basic_shape, action_dim), jnp.float32)

    print_dict_info(data)
    
    return data


class Trainer(TrainerBase):
    def add_attributes(self):
        self.imaginary_theta = self.model.theta

    def build_optimizers(self):
        theta = self.model.theta.copy()
        theta.pop('imaginary')
        self.opts.theta, self.params.theta = optimizer.build_optimizer(
            params=theta, 
            **self.config.theta_opt, 
            name='theta'
        )
        self.imaginary_opt_state = self.params.theta

    def compile_train(self):
        self.jit_train = jax.jit(
            self.theta_train, 
        )
        # self.haiku_tabulate()
    
    def train(self, data):
        self.rng, train_rng = jax.random.split(self.rng)
        epoch_rngs = jax.random.split(train_rng, self.config.n_epochs)

        theta = self.model.theta.copy()
        is_imaginary = theta.pop('imaginary')
        assert is_imaginary == False, is_imaginary
        for erng in epoch_rngs:
            rngs = jax.random.split(erng, self.config.n_mbs)
            for rng in rngs:
                with Timer('plain_train'):
                    theta, self.params.theta, stats = \
                        self.jit_train(
                            theta, 
                            rng=rng, 
                            opt_state=self.params.theta, 
                            data=data, 
                        )
        self.model.set_weights(theta)

        with Timer('stats sampling'):
            stats = sample_stats(
                stats, 
                max_record_size=100, 
            )

        return stats        

    def imaginary_train(self, data):
        self.rng, train_rng = jax.random.split(self.rng)
        epoch_rngs = jax.random.split(train_rng, self.config.n_epochs)

        theta = self.model.imaginary_params.copy()
        is_imaginary = theta.pop('imaginary')
        assert is_imaginary == True, is_imaginary
        opt_state = self.imaginary_opt_state
        for erng in epoch_rngs:
            rngs = jax.random.split(erng, self.config.n_mbs)
            for rng in rngs:
                with Timer('plain_train'):
                    theta, opt_state, _ = \
                        self.jit_train(
                            theta, 
                            rng=rng, 
                            opt_state=opt_state, 
                            data=data, 
                        )
        
        for k, v in theta.items():
            self.model.imaginary_params[k] = v
        self.imaginary_opt_state = opt_state

    def sync_imaginary_params(self):
        self.model.sync_imaginary_params()
        self.imaginary_opt_state = self.params.theta

    def get_theta_params(self):
        weights = {
            'model': self.model.theta, 
            'opt': self.params.theta
        }
        return weights
    
    def set_theta_params(self, weights):
        self.model.set_weights(weights['model'])
        self.params.theta = weights['opt']

    def theta_train(
        self, 
        theta, 
        rng, 
        opt_state, 
        data, 
    ):
        do_logging('train is traced', backtrack=4)
        theta, opt_state, stats = optimizer.optimize(
            self.loss.loss, 
            theta, 
            opt_state, 
            kwargs={
                'rng': rng, 
                'data': data, 
            }, 
            opt=self.opts.theta, 
            name='train/theta'
        )
        stats.update({f'data/{k}': lax.stop_gradient(v) 
            for k, v in data.items() if v is not None})

        return theta, opt_state, stats

    def haiku_tabulate(self, data=None):
        rng = jax.random.PRNGKey(0)
        if data is None:
            data = construct_fake_data(self.env_stats, 0)
        print(hk.experimental.tabulate(self.theta_train)(
            self.model.theta, rng, self.params.theta, data
        ))


create_trainer = partial(create_trainer,
    name='zero', trainer_cls=Trainer
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
