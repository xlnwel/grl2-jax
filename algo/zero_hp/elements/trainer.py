from functools import partial
import numpy as np
import jax
from jax import lax, random
import jax.numpy as jnp
import haiku as hk
import chex
import optax

from core.elements.trainer import TrainerBase, create_trainer
from core import optimizer
from core.log import do_logging
from core.typing import AttrDict, dict2AttrDict
from jax_tools import jax_utils
from tools.display import print_dict_info
from tools.timer import Timer
from .utils import compute_inner_steps


def construct_fake_data(env_stats, aid, meta):
    b = 32
    s = 4
    u = len(env_stats.aid2uids[aid])
    shapes = env_stats.obs_shape[aid]
    dtypes = env_stats.obs_dtype[aid]
    action_dim = env_stats.action_dim[aid]
    if meta:
        n = 2
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
        self.config = compute_inner_steps(self.config)
        self._use_meta = self.config.inner_steps is not None
        self._step = 0
        assert self.config.msmg_type in ('avg', 'wavg' 'last'), self.config.msmg_type

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
        rng = jax.random.PRNGKey(0)
        if self._use_meta:
            data = construct_fake_data(self.env_stats, 0, True)
            print(hk.experimental.tabulate(self.blo_train)(
                self.model.eta, self.model.theta, 
                rng, self.params, data
            ))
        else:
            data = construct_fake_data(self.env_stats, 0, False)
            print(hk.experimental.tabulate(self.theta_train)(
                self.model.theta, self.model.eta, 
                rng, self.params.theta, data
            ))
    

    def post_init(self):
        self.old_theta_params = self.get_theta_params()
        self.new_theta_params = self.get_theta_params()

    def train(self, data):
        self._step += 1
        do_meta_step = self._use_meta and \
            self._step % (self.config.inner_steps + self.config.extra_meta_step) == 0
        self.rng, train_rng = random.split(self.rng)

        if do_meta_step:
            self.new_theta_params = self.get_theta_params()
            with Timer('meta_train'):
                eta, theta, self.params, stats = \
                    self.jit_blo_train(
                        eta=self.model.eta, 
                        theta=self.old_theta_params, 
                        target_theta=self.new_theta_params, 
                        rng=train_rng, 
                        opt_params=self.params, 
                        data=data
                    )
            self.model.set_weights(eta)
            self.set_theta_params(self.new_theta_params)
            self.old_theta_params = self.new_theta_params
        else:
            use_meta = self._use_meta
            with Timer('plain_train'):
                theta, self.params.theta, stats = \
                    self.jit_train(
                        theta=self.model.theta, 
                        eta=self.model.eta, 
                        rng=train_rng, 
                        opt_state=self.params.theta, 
                        data=data, 
                        use_meta=use_meta, 
                    )
            assert self.params.theta is not None, self.params.theta
            self.model.set_weights(theta)

        if not self._use_meta or do_meta_step:
            with Timer('stats sampling'):
                stats = sample_stats(
                    stats, 
                    max_record_size=100, 
                )
        else:
            stats = {}

        return stats        

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
        eta, 
        rng, 
        opt_state, 
        data, 
        use_meta=False, 
        use_dice=False, 
        name='train/theta'
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
                    'use_dice': use_dice
                }, 
                opt=self.opts.theta, 
                name=name
            )
        if not use_meta:
            stats.update({f'data/{k}': lax.stop_gradient(v) 
                for k, v in data.items() if v is not None})
            stats['theta/norm'] = optax.global_norm(theta)

        return theta, opt_state, stats

    def blo_train(self, eta, theta, target_theta, rng, opt_params, data):
        rngs = random.split(rng, 3)
        grads, ((theta, opt_params.theta), stats) = \
            optimizer.compute_meta_gradients(
                self.eta_loss, 
                eta, 
                kwargs={
                    'theta': theta, 
                    'target_theta': target_theta, 
                    'rng': rngs[0], 
                    'theta_opt_state': opt_params.theta, 
                    'data': data, 
                }, 
                name='eta', 
                by_part=True
            )
        updates = AttrDict()
        updates.meta_params, opt_params.meta_params, stats = \
            optimizer.compute_updates(
                grads.meta_params, opt_params.meta_params, 
                self.opts.meta_params, stats, 
                name='eta/params'
            )
        eta = optax.apply_updates(eta, updates)

        if self.config.extra_meta_step:
            theta, opt_params.theta, theta_stats = self.theta_train(
                theta, eta, rngs[1], opt_params.theta, 
                data.slice(self.config.K), name='theta', 
                use_meta=True, use_dice=False
            )
            stats.update(theta_stats)

        stats.update({f'data/{k}': v 
            for k, v in data.items() if v is not None})
        stats['theta/norm'] = optax.global_norm(theta)
        stats['eta/norm'] = optax.global_norm(eta)

        return eta, theta, opt_params, stats

    @partial(jax.jit, static_argnums=0)
    def eta_loss(
        self, 
        eta, 
        theta, 
        target_theta, 
        rng, 
        theta_opt_state, 
        phi_opt_state, 
        data
    ):
        assert self.config.msmg_type in ['last', 'avg', 'wavg'], self.config.msmg_type
        rngs = random.split(rng, 2)

        krngs = random.split(rngs[0], self.config.K)
        data_slices = [data.slice(i) 
            for i in range(self.config.K+self.config.extra_meta_step)]
        eta_losses = []
        for i in range(self.config.K):
            theta_rng, eta_rng = random.split(krngs[i], 2)
            data_slice = data_slices[i]
            theta, theta_opt_state, stats = self.theta_train(
                theta, eta, theta_rng, theta_opt_state, 
                data_slice, name='theta', 
                use_meta=True, use_dice=True
            )
            if self.config.msmg_type == 'avg' or self.config.msmg_type == 'wavg':
                eta_loss, eta_stats = self.loss.eta_loss(
                    theta, 
                    target_theta, 
                    eta_rng, 
                    data_slices[i+self.config.extra_meta_step], 
                    name='eta'
                )
                eta_losses.append(eta_loss)
        if self.config.msmg_type == 'last':
            last_i = self.config.K + self.config.extra_meta_step - 1
            eta_loss, eta_stats = self.loss.eta_loss(
                theta, 
                target_theta, 
                rngs[1], 
                data_slices[last_i], 
                name='eta'
            )
        if eta_losses != []:
            assert self.config.msmg_type in ['avg', 'wavg'], self.config.msmg_type
            eta_losses.append(eta_loss)
            if self.config.msmg_type == 'avg':
                eta_loss = jnp.mean(jnp.stack(eta_losses))
            elif self.config.msmg_type == 'wavg':
                eta_loss = 0
                kappa = self.config.kappa_weight
                for loss in reversed(eta_losses):
                    eta_loss = eta_loss + kappa * loss
                    kappa = kappa * self.config.kappa
        else:
            assert self.config.msmg_type == 'last', self.config.msmg_type
        
        stats.update(eta_stats)

        return eta_loss, ((theta, theta_opt_state, phi_opt_state), stats)


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
