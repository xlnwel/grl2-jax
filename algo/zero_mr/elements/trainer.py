from functools import partial
import jax
import jax.numpy as jnp
import chex
import optax

from core.elements.trainer import Trainer as TrainerBase, create_trainer
from core import optimizer
from core.typing import AttrDict, dict2AttrDict
from jax_tools import jax_utils
from tools.display import print_dict_info
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
    data.global_state = data.obs
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
        assert self.config.msmg_type in ('avg', 'last'), self.config.msmg_type

    def build_optimizers(self):
        self.opts.theta, self.params.theta = optimizer.build_optimizer(
            params=self.model.theta, 
            **self.config.rl_opt, 
            name='theta'
        )
        if self._use_meta:
            self.opts.meta_reward, self.params.meta_reward = \
                optimizer.build_optimizer(
                    params=self.model.params.meta_reward, 
                    **self.config.meta_reward_opt, 
                    name='meta_reward'
                )
            self.opts.meta_params, self.params.meta_params = \
                optimizer.build_optimizer(
                    params=self.model.params.meta_params, 
                    **self.config.meta_params_opt, 
                    name='meta_params'
                )

    def jit_train(self):
        self.train = jax.jit(
            self.raw_train, 
            static_argnames=('use_meta', 'use_dice')
        )
        self.meta_train = jax.jit(
            chex.assert_max_traces(self.raw_meta_train, n=1), 
        )

    def get_rl_weights(self):
        weights = {
            'model': self.model.theta, 
            'opt': self.params.theta
        }
        return weights
    
    def set_rl_weights(self, weights):
        self.model.set_weights(weights['model'])
        self.params.theta = weights['opt']

    def raw_train(self, theta, eta, opt_state, data, use_meta=False, use_dice=False):
        if data.rl_reward is None:
            _, _, _, data.rl_reward = \
                self._compute_rl_reward(eta, data, axis=1)
            data.rl_discount, data.rl_reset = \
                self._compute_rl_discount(
                    data.discount, data.event, data.next_event, data.reset)
        theta, opt_state, stats = optimizer.optimize(
            self.loss.loss, 
            theta, 
            opt_state, 
            kwargs={
                'eta': eta, 
                'data': data, 
                'use_meta': use_meta, 
                'use_dice': use_dice
            }, 
            opt=self.opts.theta, 
            name='train/theta'
        )

        return theta, opt_state, stats
    
    def meta_loss(self, eta, theta, inner_opt_state, data):
        meta_reward_out, data.meta_reward, trans_reward, data.rl_reward = \
            self._compute_rl_reward(eta, data, axis=2)
        data.rl_discount, data.rl_reset = self._compute_rl_discount(
            data.discount, data.event, data.next_event, data.reset)
        for i in range(self.config.K):
            theta, inner_opt_state, stats = self.raw_train(
                theta, eta, inner_opt_state, data.slice(i), 
                use_meta=True, use_dice=True)
        meta_loss, meta_stats = self.loss.outer_loss(
            theta, data.slice(self.config.K))
        
        stats.update(meta_stats)
        stats.meta_reward_out = meta_reward_out
        stats.trans_reward = trans_reward
        return meta_loss, (stats, inner_opt_state)

    def raw_meta_train(self, eta, theta, opt_params, data):
        grads, (stats, opt_params.theta) = optimizer.compute_meta_gradients(
            self.meta_loss, 
            eta, 
            kwargs={
                'theta': theta, 
                'inner_opt_state': opt_params.theta, 
                'data': data
            }, 
            name='meta', 
            by_part=True
        )
        updates = AttrDict()
        updates.meta_reward, opt_params.meta_reward, stats = \
            optimizer.compute_updates(
                grads.meta_reward, opt_params.meta_reward, 
                self.opts.meta_reward, stats, 'meta_reward'
            )
        updates.meta_params, opt_params.meta_params, stats = \
            optimizer.compute_updates(
                grads.meta_params, opt_params.meta_params, 
                self.opts.meta_params, stats, 'meta_params'
            )
        eta = optax.apply_updates(eta, updates)

        return eta, theta, opt_params, stats

    def _compute_rl_reward(self, eta, data, axis):
        if self.config.K:
            [idx, event, hidden_state], [next_idx, next_event, next_hidden_state] = \
                jax_utils.split_data(
                    [data.idx, data.event, data.hidden_state], 
                    [data.next_idx, data.next_event, data.next_hidden_state], 
                    axis=axis
                )
            x, meta_reward, trans_reward = self.model.compute_meta_reward(
                eta, 
                hidden_state, 
                next_hidden_state, 
                data.action, 
                idx=idx, 
                next_idx=next_idx, 
                event=event, 
                next_event=next_event
            )
            if self.config.rl_reward == 'meta':
                rl_reward = trans_reward
            elif self.config.rl_reward == 'sum':
                rl_reward = data.reward + trans_reward
            elif self.config.rl_reward == 'interpolated':
                reward_coef = self.model.meta.meta('reward_coef', inner=True)
                rl_reward = reward_coef * data.reward + (1 - reward_coef) * meta_reward
            else:
                raise ValueError(f"Unknown rl reward type: {self.config['rl_reward']}")
            return x, meta_reward, trans_reward, rl_reward
        else:
            return None, None, None, data.reward

    def _compute_rl_discount(self, discount, event, next_event, reset):
        if event is not None and self.config.event_done:
            if reset is not None:
                discount = 1 - reset
            event, next_event = jax_utils.split_data(event, next_event)
            event_idx = jnp.argmax(event, -1)
            next_event_idx = jnp.argmax(next_event, -1)
            rl_discount = jnp.asarray(event_idx == next_event_idx, jnp.float32)
            rl_discount = jnp.where(jnp.array(discount, bool), rl_discount, discount)
            rl_reset = None
        else:
            rl_discount = discount
            rl_reset = reset
        return rl_discount, rl_reset


create_trainer = partial(create_trainer,
    name='zero', trainer_cls=Trainer
)


if __name__ == '__main__':
    import haiku as hk
    from tools.yaml_op import load_config
    from env.func import create_env
    from .model import create_model
    from .loss import create_loss
    from tools.display import pwc
    config = load_config('algo/zero_mr/configs/magw_a2c')
    
    env = create_env(config.env)
    model = create_model(config.model, env.stats())
    loss = create_loss(config.loss, model)
    trainer = create_trainer(config.trainer, env.stats(), loss)
    data = construct_fake_data(env.stats(), 0, False)
    pwc(hk.experimental.tabulate(trainer.train)(
        model.theta, model.eta, trainer.params.theta, data), color='yellow')
    # data = construct_fake_data(env.stats(), 0, True)
    # pwc(hk.experimental.tabulate(trainer.raw_meta_train)(
    #     model.eta, model.theta, trainer.params, data), color='yellow')
