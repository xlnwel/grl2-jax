import os
import math
import logging
import jax
from jax import lax, nn, random
import jax.numpy as jnp
import haiku as hk

from core.log import do_logging
from core.elements.model import Model as ModelBase
from core.typing import dict2AttrDict
from nn.func import create_network
from tools.file import source_file
from jax_tools import jax_dist
from .utils import compute_inner_steps, get_hx

# register ppo-related networks 
source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))
logger = logging.getLogger(__name__)


def construct_fake_data(env_stats, aid):
    basic_shape = (1, len(env_stats.aid2uids[aid]))
    shapes = env_stats.obs_shape[aid]
    dtypes = env_stats.obs_dtype[aid]
    action_dim = env_stats.action_dim[aid]
    data = {k: jnp.zeros((*basic_shape, *v), dtypes[k]) 
        for k, v in shapes.items()}
    data = dict2AttrDict(data)
    data.setdefault('global_state', data.obs)
    data.setdefault('hidden_state', data.obs)
    data.action = jnp.zeros((*basic_shape, action_dim), jnp.float32)
    data.hx = get_hx(data.sid, data.idx, data.event)

    return data

class Model(ModelBase):
    def build_nets(self):
        def build_fn(rng, *args, name, **kwargs):
            def build_net(*args, **kwargs):
                config = dict2AttrDict(self.config, to_copy=True)
                net = create_network(config[name], name)
                return net(*args, **kwargs)
            net = hk.transform(build_net)
            return net.init(rng, *args, **kwargs), net.apply
        aid = self.config.get('aid', 0)
        self.is_action_discrete = self.env_stats.is_action_discrete[aid]
        data = construct_fake_data(self.env_stats, aid)

        rngs = random.split(self.rng, 5)
        self.params.policy, self.modules.policy = build_fn(
            rngs[0], data.obs, data.hx, data.action_mask, name='policy')
        self.params.value, self.modules.value = build_fn(
            rngs[1], data.global_state, data.hx, name='value')
        ov_x = data.hidden_state if self.config.joint_objective else data.global_state
        ov_hx = data.sid if self.config.joint_objective else get_hx(data.sid, data.idx)
        self.params.outer_value, self.modules.outer_value = build_fn(
            rngs[2], ov_x, ov_hx, name='outer_value')
        action = None if self.config.meta_reward_type == 'shaping' else data.action
        self.params.meta_reward, self.modules.meta_reward = build_fn(
            rngs[3], data.hidden_state, action, get_hx(data.idx, data.event), 
            name='meta_reward')
        self.params.meta_params, self.modules.meta_params = build_fn(
            rngs[4], True, name='meta_params')

    @property
    def theta(self):
        return self.params.subdict('policy', 'value')

    @property
    def phi(self):
        return self.params.subdict('outer_value')
    
    @property
    def eta(self):
        return self.params.subdict('meta_reward', 'meta_params')
    
    @property
    def eta_reward(self):
        return self.params.subdict('meta_reward')
    
    @property
    def eta_params(self):
        return self.params.subdict('meta_params')
    
    def raw_action(
        self, 
        params, 
        rng, 
        data, 
        evaluation=False, 
    ):
        data.hx = self.build_hx(data)
        rngs = random.split(rng, 3)
        act_dist = self.policy_dist(params.policy, rngs[0], data, evaluation)
        action = act_dist.sample(rng=rngs[1])

        if self.is_action_discrete:
            pi = nn.softmax(act_dist.logits)
            stats = {'mu': pi}
        else:
            mean = act_dist.mean()
            std = lax.exp(act_dist.logstd)
            stats = {
                'mu_mean': mean,
                'mu_std': jnp.broadcast_arrays(mean, std), 
            }

        state = data.state
        if evaluation:
            data.action = action
            stats = self.compute_eval_terms(
                params, 
                rngs[2], 
                data, 
            )
            return action, stats, state
        else:
            logprob = act_dist.log_prob(action)
            value = self.modules.value(
                params.value, 
                rngs[2], 
                data.get('global_state', data.obs), 
                hx=data.hx
            )
            stats.update({'mu_logprob': logprob, 'value': value})

            return action, stats, state

    def policy_dist(self, params, rng, data, evaluation):
        act_out = self.modules.policy(
            params, 
            rng, 
            data.obs, 
            hx=data.hx, 
            action_mask=data.action_mask, 
        )
        if self.is_action_discrete:
            if evaluation and self.config.eval_act_temp:
                act_out = act_out / self.config.eval_act_temp
            dist = jax_dist.Categorical(act_out)
        else:
            mu, logstd = act_out
            if evaluation and self.config.eval_act_temp:
                logstd = logstd + math.log(self.config.eval_act_temp)
            dist = jax_dist.MultivariateNormalDiag(mu, logstd)

        return dist
    
    def build_hx(self, data):
        return get_hx(data.sid, data.idx, data.event)

    def compute_eval_terms(self, params, rng, data):
        value = self.modules.value(
            params.value, 
            rng, 
            data.get('global_state', data.obs), 
            hx=data.hx
        )
        value = jnp.squeeze(value)
        if self.config.K:
            _, meta_reward, trans_reward = self.compute_meta_reward(
                params, 
                rng, 
                data.prev_hidden_state, 
                data.hidden_state, 
                data.action, 
                idx=data.prev_idx, 
                next_idx=data.idx, 
                event=data.prev_event, 
                next_event=data.event, 
                shift=True
            )
            meta_reward = jnp.squeeze(meta_reward)
            trans_reward = jnp.squeeze(trans_reward)
            stats = {
                'meta_reward': meta_reward, 
                'trans_reward': trans_reward, 
                'value': value, 
                'action': data.action
            }
        else:
            stats = {'value': value, 'action': data.action}
        return stats

    def compute_meta_reward(
        self, 
        eta, 
        rng, 
        hidden_state, 
        next_hidden_state, 
        action, 
        idx=None, 
        next_idx=None, 
        event=None, 
        next_event=None, 
        hx=None, 
        next_hx=None, 
        shift=False,    # whether to shift hidden_state/idx/event by one step. If so action is at the same step as the next stats, 
        reward=None
    ):
        if hx is None:
            hx = get_hx(idx, event)
        if next_hx is None:
            next_hx = get_hx(next_idx, next_event)

        rngs = random.split(rng)
        meta_params = self.modules.meta_params(
            eta.meta_params, rngs[0], inner=True)
        if self.config.meta_reward_type == 'shaping':
            x, phi = self.modules.meta_reward(
                eta.meta_reward, rngs[1], hidden_state, hx=hx)
            _, next_phi = self.modules.meta_reward(
                eta.meta_reward, rngs[1], next_hidden_state, hx=next_hx)
            meta_reward = self.config.gamma * next_phi - phi
        elif self.config.meta_reward_type == 'intrinsic':
            x = next_hidden_state if shift else hidden_state
            x, meta_reward = self.modules.meta_reward(
                eta.meta_reward, rngs[1], x, action, hx=next_hx if shift else hx)
        else:
            raise ValueError(f"Unknown meta rewared type: {self.config.meta_reward_type}")
        
        reward_scale = meta_params.reward_scale
        reward_bias = meta_params.reward_bias
        trans_reward = reward_scale * meta_reward + reward_bias

        if reward is None:
            return x, meta_reward, trans_reward
        else:
            if self.config.rl_reward == 'meta':
                rl_reward = trans_reward
            elif self.config.rl_reward == 'sum':
                rl_reward = reward + trans_reward
            elif self.config.rl_reward == 'interpolated':
                reward_coef = meta_params.reward_coef
                rl_reward = reward_coef * reward + (1 - reward_coef) * meta_reward
            else:
                raise ValueError(f"Unknown rl reward type: {self.config['rl_reward']}")
            return x, meta_reward, trans_reward, rl_reward

def setup_config_from_envstats(config, env_stats):
    if 'aid' in config:
        aid = config['aid']
        config.policy.action_dim = env_stats.action_dim[aid]
        config.policy.is_action_discrete = env_stats.is_action_discrete[aid]
        config.meta_reward.action_dim = env_stats.action_dim[aid]
        config.meta_params.reward_scale.shape = (len(env_stats.aid2uids[aid]),)
        config.meta_params.reward_bias.shape = (len(env_stats.aid2uids[aid]),)
        config.meta_params.reward_coef.shape = (len(env_stats.aid2uids[aid]),)
    else:
        config.policy.action_dim = env_stats.action_dim
        config.policy.is_action_discrete = env_stats.is_action_discrete
        config.policy.action_low = env_stats.action_low
        config.policy.action_high = env_stats.action_high
        config.meta_reward.action_dim = env_stats.action_dim
        config.meta_params.reward_scale.shape = (env_stats.n_units,)
        config.meta_params.reward_bias.shape = (env_stats.n_units,)
        config.meta_params.reward_coef.shape = (env_stats.n_units,)

    return config


def create_model(
    config, 
    env_stats, 
    name='zero', 
    **kwargs
): 
    config = setup_config_from_envstats(config, env_stats)
    config = compute_inner_steps(config)

    return Model(
        config=config, 
        env_stats=env_stats, 
        name=name,
        **kwargs
    )


if __name__ == '__main__':
    from tools.yaml_op import load_config
    from env.func import create_env
    from tools.display import pwc
    config = load_config('algo/zero_mr/configs/magw_a2c')
    
    env = create_env(config.env)
    model = create_model(config.model, env.stats())
    data = construct_fake_data(env.stats(), 0)
    print(model.action(model.params, data))
    pwc(hk.experimental.tabulate(model.raw_action)(model.params, data), color='yellow')
