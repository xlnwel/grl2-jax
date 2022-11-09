import collections
from functools import partial
import numpy as np
import jax
from jax import lax, random, tree_util
import jax.numpy as jnp
import haiku as hk
import chex
import optax

from core.elements.trainer import TrainerBase, create_trainer
from core import optimizer
from core.log import do_logging
from core.typing import AttrDict, dict2AttrDict
from jax_tools.jax_utils import compute_norms
from tools.display import print_dict_info
from tools.timer import Timer
from tools.utils import flatten_dict
from .utils import compute_inner_steps


def compute_debug_stats(model, params, rng, data):
    act_dist, stats = model.forward(params, rng, data)
    entropy = act_dist.entropy()
    kl = (stats.pi_logprob - data.mu_logprob)
    entropy = np.array(entropy)
    kl = np.array(kl)
    result = {
        'entropy': entropy.mean(), 
        'entropy_max': entropy.max(), 
        'kl': kl.mean(), 
        'kl_max': kl.max(), 
    }
    return result


def construct_fake_data(config, env_stats, aid, meta):
    b = 32
    s = 4
    u = len(env_stats.aid2uids[aid])
    shapes = env_stats.obs_shape[aid]
    dtypes = env_stats.obs_dtype[aid]
    action_dim = env_stats.action_dim[aid]
    is_action_discrete = env_stats.is_action_discrete[aid]
    if meta:
        n = config.inner_steps
        basic_shape = (n, b, s, u)
        data = {k: jnp.zeros((n, b, s+1, u, *v), dtypes[k]) 
            for k, v in shapes.items()}
    else:
        basic_shape = (b, s, u)
        data = {k: jnp.zeros((b, s+1, u, *v), dtypes[k]) 
            for k, v in shapes.items()}
    data = dict2AttrDict(data)
    data.setdefault('global_state', data.obs)
    data.setdefault('hidden_state', data.obs)
    data.value = jnp.zeros(basic_shape, jnp.float32)
    data.reward = jnp.zeros(basic_shape, jnp.float32)
    data.discount = jnp.zeros(basic_shape, jnp.float32)
    data.reset = jnp.zeros(basic_shape, jnp.float32)
    data.mu_logprob = jnp.zeros(basic_shape, jnp.float32)
    if is_action_discrete:
        data.action = jnp.zeros(basic_shape, jnp.int32)
        data.mu_logits = jnp.zeros((*basic_shape, action_dim), jnp.float32)
    else:
        data.action = jnp.zeros((*basic_shape, action_dim), jnp.int32)
        data.mu_mean = jnp.zeros((*basic_shape, action_dim), jnp.float32)
        data.mu_std = jnp.zeros((*basic_shape, action_dim), jnp.float32)

    print_dict_info(data)

    if meta:
        data = [data.slice(i) for i in range(config.K+config.L)]
        for d in data:
            d.meta_param_stats = AttrDict({
                'entropy_coef': np.zeros((1, 1))
            })
    else:
        data.meta_param_stats = AttrDict({
            'entropy_coef': np.zeros((1, 1))
        })

    return data


class Trainer(TrainerBase):
    def add_attributes(self):
        self.config = compute_inner_steps(self.config)
        self._debug = self.config.get('debug', True)
        self._use_meta = self.config.inner_steps is not None
        self._step = 0

        self.reward_history = collections.deque(maxlen=10)

    def build_optimizers(self):
        self.opts.theta, self.params.theta = optimizer.build_optimizer(
            params=self.model.theta, 
            **self.config.theta_opt, 
            name='theta'
        )
        if self._use_meta:
            self.opts.meta_params, self.params.meta_params = \
                optimizer.build_optimizer(
                    params=self.model.params.meta_params, 
                    **self.config.meta_params_opt, 
                    name='meta_params'
                )

    def compile_train(self):
        self.jit_train = jax.jit(
            self.theta_train, 
            static_argnames=('use_meta', 'use_dice', 'name')
        )
        self.jit_blo_train = jax.jit(
            self.blo_train, 
        )
        # self.haiku_tabulate()

    def post_init(self):
        self.old_theta_params = self.get_theta_params()
        chex.assert_trees_all_close(self.model.theta, self.old_theta_params.model)

    def train(self, data):
        self._step += 1
        do_meta_step = self._use_meta and \
            self._step % self.config.inner_steps == 0
        self.rng, train_rng = random.split(self.rng)
        # print('train step', self._step, do_meta_step)

        if do_meta_step:
            data_slices = self.prepare_data_for_meta_learning(data)
            train_rng = random.split(train_rng, 1)[0]
            with Timer('meta_train'):
                opt_params = self.params.copy()
                opt_params.theta = self.old_theta_params.opt
                eta, next_theta, self.params.meta_params, stats = \
                    self.jit_blo_train(
                        eta=self.model.eta, 
                        theta=self.old_theta_params.model, 
                        target_theta=self.model.theta, 
                        rng=train_rng, 
                        opt_params=opt_params, 
                        data_slices=data_slices
                    )
            self.model.set_weights(eta)
            # chex.assert_trees_all_close(self.model.theta, next_theta)
            self.model.set_weights(next_theta)
            # debug_stats = compute_debug_stats(
            #     self.model, self.model.theta, train_rng, data_slices[-1])
            # do_logging(f'meta update: {debug_stats}')
        use_meta = self._use_meta
        rl_data = data.slice(-1) if do_meta_step else data
        with Timer('plain_train'):
            theta, self.params.theta, theta_stats = \
                self.jit_train(
                    theta=self.model.theta, 
                    eta=self.model.eta, 
                    rng=train_rng, 
                    opt_state=self.params.theta, 
                    data=rl_data, 
                    use_meta=use_meta, 
                    use_dice=False
                )
        self.model.set_weights(theta)
        if do_meta_step:
            self.old_theta_params = self.get_theta_params()
        else:
            stats = theta_stats

        if self._use_meta:
            return_stats = self._step % (3 * self.config.inner_steps) == 0
        else:
            return_stats = self._step % 50 == 0
        if return_stats:
            data.reward_stats = data.meta_param_stats.entropy_coef
            with Timer('stats_sampling'):
                stats = sample_stats(
                    stats, 
                    data, 
                    max_record_size=10, 
                )
        else:
            stats = {}

        return stats

    def get_theta_params(self):
        weights = dict2AttrDict({
            'model': self.model.theta, 
            'opt': self.params.theta
        }, shallow=True)
        return weights
    
    def set_theta_params(self, weights):
        self.model.set_weights(weights.model)
        self.params.theta = weights.opt

    def theta_train(
        self, 
        theta, 
        eta, 
        rng, 
        opt_state, 
        data, 
        use_meta=False, 
        use_dice=False, 
        name='theta'
    ):
        rngs = random.split(rng, 2)
        n_epochs = self.config.get('n_epochs', 1)
        rngs = random.split(rngs[1], n_epochs)
        for rng in rngs:
            theta, opt_state, stats = optimizer.optimize(
                self.loss.loss, 
                theta, 
                opt_state, 
                kwargs={
                    'eta': eta, 
                    'rng': rng, 
                    'data': data, 
                    'use_meta': use_meta, 
                    'use_dice': use_dice, 
                    'name': name
                }, 
                opt=self.opts.theta, 
                name=name, 
                debug=self._debug
            )
        
        if not use_meta:
            stats['theta/var/norm'] = optax.global_norm(theta)
            if self._debug:
                theta_norm = compute_norms(theta)
                stats.update(flatten_dict(theta_norm, suffix='var/norm'))

        return theta, opt_state, stats

    def blo_train(
        self, 
        eta, 
        theta, 
        target_theta, 
        rng, 
        opt_params, 
        data_slices
    ):
        grads, ((next_theta, opt_params.theta), stats) = \
            optimizer.compute_meta_gradients(
                self.eta_loss, 
                eta, 
                kwargs={
                    'theta': theta, 
                    'target_theta': target_theta, 
                    'rng': rng, 
                    'theta_opt_state': opt_params.theta, 
                    'data_slices': data_slices, 
                }, 
                name='eta', 
                debug=self._debug
            )
        updates = AttrDict()
        updates.meta_params, opt_params.meta_params, stats = \
            optimizer.compute_updates(
                grads.meta_params, opt_params.meta_params, 
                self.opts.meta_params, stats, 
                name='eta', 
                debug=self._debug, 
            )
        eta = optax.apply_updates(eta, updates)

        stats['theta/var/norm'] = optax.global_norm(theta)
        stats['eta/var/norm'] = optax.global_norm(eta)
        if self._debug:
            theta_norm = compute_norms(theta)
            stats.update(flatten_dict(theta_norm, suffix='var/norm'))
            eta_norm = compute_norms(eta)
            stats.update(flatten_dict(eta_norm, suffix='var/norm'))

        return eta, next_theta, opt_params.meta_params, stats

    @partial(jax.jit, static_argnums=0)
    def eta_loss(
        self, 
        eta, 
        theta, 
        target_theta, 
        rng, 
        theta_opt_state, 
        data_slices
    ):
        rngs = random.split(rng, self.config.K+self.config.L+1)
        for i in range(self.config.K):
            theta, theta_opt_state, stats = self.theta_train(
                theta, eta, rngs[i], theta_opt_state, 
                data_slices[i], name='theta', 
                use_meta=True, use_dice=True
            )
        if self.config.L:
            next_theta = lax.stop_gradient(theta)
            for i in range(self.config.L-1):
                idx = self.config.K + i
                next_theta, theta_opt_state, next_stats = self.theta_train(
                    next_theta, eta, rngs[idx], theta_opt_state, 
                    data_slices[idx], name='next_theta', 
                    use_meta=True, use_dice=True
                )
            target_theta, theta_opt_state, target_stats = self.theta_train(
                next_theta, eta, rngs[-2], theta_opt_state, 
                data_slices[-1], name='target_theta', 
                use_meta=False, use_dice=False
            )
            if self.config.L > 1:
                stats.update(next_stats)
            stats.update(target_stats)
        else:
            assert self.config.meta_type == 'plain', self.config.meta_type
            next_theta = theta
            target_theta = theta

        eta_loss, eta_stats = self.loss.eta_loss(
            theta, 
            target_theta, 
            rngs[-1], 
            data_slices[-1], 
            name='eta'
        )

        stats.update(eta_stats)

        return eta_loss, ((next_theta, theta_opt_state), stats)

    def prepare_data_for_meta_learning(self, data):
        data = [data.slice(i) for i in range(self.config.K+self.config.L)]
        return data

    def compute_kl(self, theta, target_theta, rng, data):
        _, theta_stats = self.model.forward(theta, rng, data)
        _, target_stats = self.model.forward(target_theta, rng, data)
        log_ratio = theta_stats.pi_logprob - target_stats.pi_logprob
        approx_kl = .5 * jnp.mean((log_ratio)**2)
        return approx_kl

    def haiku_tabulate(self, data=None):
        rng = jax.random.PRNGKey(0)
        if self._use_meta:
            if data is None:
                data = construct_fake_data(self.config, self.env_stats, 0, True)
            print(hk.experimental.tabulate(self.blo_train)(
                self.model.eta, self.model.theta, self.model.theta, 
                rng, self.params, data
            ))
        else:
            if data is None:
                data = construct_fake_data(self.config, self.env_stats, 0, False)
            print(hk.experimental.tabulate(self.theta_train)(
                self.model.theta, self.model.eta, 
                rng, self.params.theta, data
            ))


create_trainer = partial(create_trainer,
    name='zero', trainer_cls=Trainer
)


def sample_stats(stats, data, max_record_size=10):
    # we only sample a small amount of data to reduce the cost
    batch_dims = data['reward'].ndim
    sampled_stats = _sample_stats(data, batch_dims, max_record_size, prefix='data')
    sampled_stats.update(_sample_stats(stats, 3, max_record_size))
    return sampled_stats


def _sample_stats(stats, batch_dims, max_record_size, prefix=None):
    sampled_stats = {}
    for k, v in stats.items():
        k = k if prefix is None else f'{prefix}/{k}'
        if isinstance(v, (int, float)):
            sampled_stats[k] = v
        elif isinstance(v, (np.ndarray, jnp.DeviceArray)):
            if prefix and prefix.startswith('data'):
                v = v.reshape(-1, *v.shape[batch_dims:])
            else:
                v = v.reshape(-1)
            if v.shape[0] > max_record_size:
                idx = np.random.choice(np.arange(v.shape[0]), max_record_size)
                v = v[idx]
            sampled_stats[k] = v
        elif isinstance(v, dict):
            prefix = k
            sampled_stats.update(_sample_stats(
                v, batch_dims, max_record_size, prefix=prefix))
        else:
            do_logging(f'{k} of type({type(v)}): {v}')
            raise ValueError()
    return sampled_stats


if __name__ == '__main__':
    import haiku as hk
    from tools.yaml_op import load_config
    from env.func import create_env
    from .model import create_model
    from .loss import create_loss
    from core.log import pwc
    config = load_config('algo/zero_mr/configs/magw_a2c')
    
    env = create_env(config.env)
    model = create_model(config.model, env.stats())
    loss = create_loss(config.loss, model)
    trainer = create_trainer(config.trainer, env.stats(), loss)
    data = construct_fake_data(env.stats(), 0, False)
    pwc(hk.experimental.tabulate(trainer.train)(
        model.theta, model.eta, trainer.params.theta, data), color='yellow')
    # data = construct_fake_data(env.stats(), 0, True)
    # pwc(hk.experimental.tabulate(trainer.blo_train)(
    #     model.eta, model.theta, trainer.params, data), color='yellow')
