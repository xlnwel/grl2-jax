from functools import partial
import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
import haiku as hk

from core.log import do_logging, pwc
from core.elements.trainer import TrainerBase, create_trainer
from core import optimizer
from core.typing import AttrDict, dict2AttrDict
from tools.display import print_dict_info
from tools.timer import Timer
from tools.utils import flatten_dict


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
    data.next_value = jnp.zeros(basic_shape, jnp.float32)
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
    def add_attributes(self):
        self.imaginary_theta = self.model.theta
        self.indices = np.arange(self.config.n_runners * self.config.n_envs)

    def build_optimizers(self):
        theta = self.model.theta.copy()
        theta.pop('imaginary')
        self.opts.theta, self.params.theta = optimizer.build_optimizer(
            params=theta, 
            **self.config.theta_opt, 
            name='theta'
        )
        self.imaginary_opt_state = self.params.theta

    def train(self, data: AttrDict):
        theta = self.model.theta.copy()
        is_imaginary = theta.pop('imaginary')
        assert is_imaginary == False, is_imaginary
        for _ in range(self.config.n_epochs):
            np.random.shuffle(self.indices)
            indices = np.split(self.indices, self.config.n_mbs)
            for idx in indices:
                with Timer('theta_train'):
                    theta, self.params.theta, stats = \
                        self.jit_train(
                            theta, 
                            opt_state=self.params.theta, 
                            data=data.slice(idx), 
                        )
        self.model.set_weights(theta)

        data = flatten_dict({f'data/{k}': v 
            for k, v in data.items() if v is not None})
        stats.update(data)
        with Timer('stats_subsampling'):
            stats = sample_stats(
                stats, 
                max_record_size=100, 
            )
        for v in theta.values():
            stats.update(flatten_dict(
                jax.tree_util.tree_map(np.linalg.norm, v)))

        return stats

    def imaginary_train(self, data: AttrDict):
        theta = self.model.imaginary_params.copy()
        is_imaginary = theta.pop('imaginary')
        assert is_imaginary == True, is_imaginary
        opt_state = self.imaginary_opt_state
        for _ in range(self.config.n_epochs):
            with Timer('imaginary_train'):
                theta, opt_state, _ = \
                    self.jit_train(
                        theta, 
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

        return theta, opt_state, stats

    # def haiku_tabulate(self, data=None):
    #     rng = jax.random.PRNGKey(0)
    #     if data is None:
    #         data = construct_fake_data(self.env_stats, 0)
    #     theta = self.model.theta.copy()
    #     is_imaginary = theta.pop('imaginary')
    #     print(hk.experimental.tabulate(self.theta_train)(
    #         theta, rng, self.params.theta, data
    #     ))
    #     breakpoint()


create_trainer = partial(create_trainer,
    name='zero', trainer_cls=Trainer
)


def sample_stats(stats, max_record_size=10):
    # we only sample a small amount of data to reduce the cost
    stats = {k if '/' in k else f'train/{k}': 
        np.random.choice(stats[k].reshape(-1), max_record_size) 
        if isinstance(stats[k], (np.ndarray, jnp.DeviceArray)) \
            else stats[k] 
        for k in sorted(stats.keys())}
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
