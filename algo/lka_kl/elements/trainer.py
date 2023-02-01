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
from tools.rms import RunningMeanStd
from tools.timer import Timer
from tools.utils import flatten_dict, prefix_name
from jax_tools import jax_utils
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

    def compile_train(self):
        _img_jit_train = jax.jit(self.theta_train)
        _jit_train = jax.jit(self.real_theta_train)
        def img_jit_train(*args, **kwargs):
            self.rng, rng = jax.random.split(self.rng)
            return _img_jit_train(*args, rng=rng, **kwargs)
        def jit_train(*args, **kwargs):
            self.rng, rng = jax.random.split(self.rng)
            return _jit_train(*args, rng=rng, **kwargs)
        self.jit_train = jit_train
        self.jit_img_train = img_jit_train

        self.haiku_tabulate()

    def train(self, data: AttrDict):
        if self.config.n_runners * self.config.n_envs < self.config.n_mbs:
            self.indices = np.arange(self.config.n_mbs)
            data = jax_utils.tree_map(
                lambda x: jnp.reshape(x, (self.config.n_mbs, -1, *x.shape[2:])), data)

        theta = self.model.theta.copy()
        is_lookahead = theta.pop('lookahead')
        assert is_lookahead == False, is_lookahead
        for _ in range(self.config.n_epochs):
            np.random.shuffle(self.indices)
            indices = np.split(self.indices, self.config.n_mbs)
            v_target = []
            for idx in indices:
                with Timer('theta_train'):
                    d = data.slice(idx)
                    if self.config.popart:
                        d.popart_mean = self.popart.mean
                        d.popart_std = self.popart.std
                    theta, self.params.theta, stats = \
                        self.jit_train(
                            theta, 
                            opt_state=self.params.theta, 
                            data=d, 
                        )
                v_target.append(stats.v_target)
        self.model.set_weights(theta)
        if self.config.popart:
            v_target = np.concatenate(v_target)
            self.popart.update(v_target)

        data = flatten_dict({f'data/{k}': v 
            for k, v in data.items() if v is not None})
        stats = prefix_name(stats, 'train')
        stats.update(data)
        stats['popart/mean'] = self.popart.mean
        stats['popart/std'] = self.popart.std
        with Timer('stats_subsampling'):
            stats = sample_stats(
                stats, 
                max_record_size=100, 
            )
        for v in theta.values():
            stats.update(flatten_dict(
                jax.tree_util.tree_map(np.linalg.norm, v)))

        return stats
        
    def lookahead_train(self, data: AttrDict):
        # NOTE: we utilize the params
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

    def real_theta_train(
        self,
        theta,
        rng,
        opt_state,
        data,
    ):
        do_logging('train is traced', backtrack=4)
        if self.config.get('theta_opt'):
            theta, opt_state, stats = optimizer.optimize(
                self.loss.real_loss, 
                theta, 
                opt_state, 
                kwargs={
                    'rng': rng, 
                    'data': data, 
                }, 
                opt=self.opts.theta, 
                name='train/theta'
            )
        else:
            assert 0, "Not here."
            theta.value, opt_state.value, stats = optimizer.optimize(
                self.loss.value_loss, 
                theta.value, 
                opt_state.value, 
                kwargs={
                    'rng': rng, 
                    'policy_theta': theta.policy, 
                    'data': data, 
                }, 
                opt=self.opts.value, 
                name='train/value'
            )
            theta.policy, opt_state.policy, stats = optimizer.optimize(
                self.loss.policy_loss, 
                theta.policy, 
                opt_state.policy, 
                kwargs={
                    'rng': rng, 
                    'data': data, 
                    'stats': stats
                }, 
                opt=self.opts.policy, 
                name='train/policy'
            )

        return theta, opt_state, stats


    def theta_train(
        self,
        theta, 
        rng, 
        opt_state, 
        data, 
    ):
        do_logging('train is traced', backtrack=4)
        if self.config.get('theta_opt'):
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
        else:
            theta.value, opt_state.value, stats = optimizer.optimize(
                self.loss.value_loss, 
                theta.value, 
                opt_state.value, 
                kwargs={
                    'rng': rng, 
                    'policy_theta': theta.policy, 
                    'data': data, 
                }, 
                opt=self.opts.value, 
                name='train/value'
            )
            theta.policy, opt_state.policy, stats = optimizer.optimize(
                self.loss.policy_loss, 
                theta.policy, 
                opt_state.policy, 
                kwargs={
                    'rng': rng, 
                    'data': data, 
                    'stats': stats
                }, 
                opt=self.opts.policy, 
                name='train/policy'
            )

        return theta, opt_state, stats


create_trainer = partial(create_trainer,
    name='lka_kl', trainer_cls=Trainer
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
