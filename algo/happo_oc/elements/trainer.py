from functools import partial
import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
import haiku as hk

from core.ckpt.pickle import save, restore
from core.log import do_logging
from core.elements.trainer import TrainerBase, create_trainer
from core import optimizer
from core.typing import AttrDict, dict2AttrDict
from tools.display import print_dict_info
from tools.rms import RunningMeanStd
from tools.timer import Timer
from tools.utils import flatten_dict, prefix_name
from jax_tools import jax_utils


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
    def add_attributes(self):
        self.imaginary_theta = self.model.theta
        self.popart = RunningMeanStd((0, 1, 2))
        self.indices = np.arange(self.config.n_runners * self.config.n_envs)

    def build_optimizers(self):
        theta = self.model.theta.copy()
        theta.pop('imaginary')
        if self.config.get('theta_opt'):
            self.opts.theta, self.params.theta = optimizer.build_optimizer(
                params=theta, 
                **self.config.theta_opt, 
                name='theta'
            )
        else:
            self.params.theta = AttrDict()
            self.opts.policy, self.params.theta.policy = optimizer.build_optimizer(
                params=theta.policy, 
                **self.config.policy_opt, 
                name='policy'
            )
            self.opts.value, self.params.theta.value = optimizer.build_optimizer(
                params=theta.value, 
                **self.config.value_opt, 
                name='value'
            )
        self.imaginary_opt_state = self.params.theta

    def compile_train(self):
        _jit_train = jax.jit(self.theta_train)
        def jit_train(*args, **kwargs):
            self.rng, rng = jax.random.split(self.rng)
            return _jit_train(*args, rng=rng, **kwargs)
        self.jit_train = jit_train
        self.jit_img_train = jit_train

        self.haiku_tabulate()

    def train(self, data: AttrDict, teammate_log_ratio=None):
        if self.config.n_runners * self.config.n_envs < self.config.n_mbs:
            self.indices = np.arange(self.config.n_mbs)
            data = jax_utils.tree_map(
                lambda x: jnp.reshape(x, (self.config.n_mbs, -1, *x.shape[2:])), data)

        if teammate_log_ratio is None:
            # TODO: Only apply happo when n_imaginary_runs==0 
            teammate_log_ratio = 0

        theta = self.model.theta.copy()
        is_imaginary = theta.pop('imaginary')
        assert is_imaginary == False, is_imaginary
        for _ in range(self.config.n_epochs):
            np.random.shuffle(self.indices)
            indices = np.split(self.indices, self.config.n_mbs)
            v_target = []
            for idx in indices:
                with Timer('theta_train'):
                    d = data.slice(idx)
                    if isinstance(teammate_log_ratio, (float, int)):
                        t_log_ratio = teammate_log_ratio
                    else:
                        t_log_ratio = teammate_log_ratio[idx]
                    if self.config.popart:
                        d.popart_mean = self.popart.mean
                        d.popart_std = self.popart.std
                    theta, self.params.theta, stats = \
                        self.jit_train(
                            theta, 
                            opt_state=self.params.theta, 
                            data=d,
                            teammate_log_ratio=t_log_ratio,
                        )
                v_target.append(stats.v_target)
        self.model.set_weights(theta)
        if self.config.popart:
            v_target = np.concatenate(v_target)
            self.popart.update(v_target)

        raw_data = data
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

        # Accumulate the agent log ratio
        self.rng, rng = jax.random.split(self.rng)
        pi_logprob = self.model.jit_action_logprob(
            self.model.params, rng, raw_data)
        agent_log_ratio = pi_logprob - raw_data.mu_logprob
        teammate_log_ratio += agent_log_ratio
        stats['teammate_log_ratio'] = teammate_log_ratio

        return stats

    def imaginary_train(self, data: AttrDict, teammate_log_ratio=None):
        theta = self.model.imaginary_params.copy()
        is_imaginary = theta.pop('imaginary')
        assert is_imaginary == True, is_imaginary
        opt_state = self.imaginary_opt_state
        for _ in range(self.config.n_imaginary_epochs):
            np.random.shuffle(self.indices)
            indices = np.split(self.indices, self.config.n_mbs)
            for idx in indices:
                with Timer('imaginary_train'):
                    d = data.slice(idx)
                    if isinstance(teammate_log_ratio, (float, int)):
                        t_log_ratio = teammate_log_ratio
                    else:
                        t_log_ratio = teammate_log_ratio[idx]
                    if self.config.popart:
                        d.popart_mean = self.popart.mean
                        d.popart_std = self.popart.std
                    theta, opt_state, _ = \
                        self.jit_img_train(
                            theta, 
                            opt_state=opt_state, 
                            data=d,
                            teammate_log_ratio=t_log_ratio,
                        )
        for k, v in theta.items():
            self.model.imaginary_params[k] = v
        self.imaginary_opt_state = opt_state

        self.rng, rng = jax.random.split(self.rng) 
        new_mu_logprob = self.model.jit_action_logprob(
            self.model.imaginary_params, rng, data)
        agent_log_ratio = new_mu_logprob - data.mu_logprob
        return teammate_log_ratio + agent_log_ratio

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
        teammate_log_ratio,
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
                    'teammate_log_ratio': teammate_log_ratio,
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
                    'stats': stats,
                    'teammate_log_ratio': teammate_log_ratio,
                }, 
                opt=self.opts.policy, 
                name='train/policy'
            )

        return theta, opt_state, stats

    def save_optimizer(self):
        super().save_optimizer()
        self.save_popart()
    
    def restore_optimizer(self):
        super().restore_optimizer()
        self.restore_popart()

    def save(self):
        super().save()
        self.save_popart()
    
    def restore(self):
        super().restore()
        self.restore_popart()

    def get_popart_dir(self):
        path = '/'.join([self.config.root_dir, self.config.model_name])
        return path

    def save_popart(self):
        filedir = self.get_popart_dir()
        save(self.popart, filedir=filedir, filename='popart')

    def restore_popart(self):
        filedir = self.get_popart_dir()
        self.popart = restore(filedir=filedir, filename='popart', default=RunningMeanStd((0, 1, 2)))

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
    name='happo', trainer_cls=Trainer
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
