import os
import math
import logging
import jax
from jax import lax, random
import jax.numpy as jnp
import haiku as hk

from core.log import do_logging
from core.elements.model import Model as ModelBase
from core.typing import AttrDict, dict2AttrDict
from nn.func import create_network
from tools.file import source_file
from tools.utils import flatten_dict
from jax_tools import jax_dist
from .utils import compute_inner_steps, compute_policy, compute_values

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
        stats = dict2AttrDict({
            'entropy_coef': jnp.zeros((1, 1))
        })
        self.params.meta_params, self.modules.meta_params = build_fn(
            rngs[4], True, stats, name='meta_params')

    @property
    def theta(self):
        return self.params.subdict(*self.theta_keys)

    @property
    def eta(self):
        return self.params.subdict(*self.eta_keys)
    
    @property
    def theta_keys(self):
        return ['policy', 'value']

    @property
    def eta_keys(self):
        return ['meta_params']

    def raw_action(
        self, 
        params, 
        rng, 
        data, 
        evaluation=False, 
    ):
        rngs = random.split(rng, 3)
        act_dist = self.policy_dist(params.policy, rngs[0], data, evaluation)
        action = act_dist.sample(rng=rngs[1])

        if self.is_action_discrete:
            stats = {'mu_logits': act_dist.logits}
        else:
            mean = act_dist.mean()
            std = lax.exp(act_dist.logstd)
            stats = {
                'mu_mean': mean,
                'mu_std': jnp.broadcast_to(std, mean.shape), 
            }

        state = data.state
        data.setdefault('global_state', data.obs)
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
                data.global_state, 
            )
            stats.update({'mu_logprob': logprob, 'value': value})

            return action, stats, state

    def policy_dist(self, params, rng, data, evaluation):
        act_out = self.modules.policy(
            params, 
            rng, 
            data.obs, 
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
    
    def forward(self, params, rng, data):
        rngs = jax.random.split(rng, 2)
        stats = AttrDict()
        stats.value, stats.next_value = compute_values(
            self.modules.value, 
            params.value, 
            rngs[0], 
            data.global_state, 
            data.next_global_state, 
            seq_axis=1, 
        )

        act_dist, stats.pi_logprob, stats.log_ratio, stats.ratio = \
            compute_policy(
                self.modules.policy, 
                params.policy, 
                rngs[1], 
                data.obs, 
                data.next_obs, 
                data.action, 
                data.mu_logprob, 
                action_mask=data.action_mask, 
                next_action_mask=data.next_action_mask, 
                seq_axis=1, 
            )

        return act_dist, stats

    def compute_eval_terms(self, params, rng, data):
        value = self.modules.value(
            params.value, 
            rng, 
            data.global_state, 
        )
        value = jnp.squeeze(value)
        stats = {'value': value, 'action': data.action}
        return stats


def setup_config_from_envstats(config, env_stats):
    if 'aid' in config:
        aid = config['aid']
        config.policy.action_dim = env_stats.action_dim[aid]
        config.policy.is_action_discrete = env_stats.is_action_discrete[aid]
    else:
        config.policy.action_dim = env_stats.action_dim
        config.policy.is_action_discrete = env_stats.is_action_discrete
        config.policy.action_low = env_stats.action_low
        config.policy.action_high = env_stats.action_high

    return config


def setup_outer_meta_params(config):
    if config.meta_type == 'bmg' and config.inner_steps:
        config.meta_params.entropy_coef.outer = 0
    else:
        config.meta_params.entropy_coef.outer = config.meta_params.entropy_coef.default
    return config


def create_model(
    config, 
    env_stats, 
    name='zero', 
    **kwargs
): 
    config = setup_config_from_envstats(config, env_stats)
    config = compute_inner_steps(config)
    config = setup_outer_meta_params(config)

    return Model(
        config=config, 
        env_stats=env_stats, 
        name=name,
        **kwargs
    )


if __name__ == '__main__':
    os.environ['XLA_FLAGS'] = "--xla_gpu_force_compilation_parallelism=1"

    from tools.yaml_op import load_config
    from env.func import create_env
    config = load_config('algo/bmg/configs/tc')
    
    env = create_env(config.env)
    model = create_model(config.model, env.stats())
    data = construct_fake_data(env.stats(), 0)
    print(model.raw_action(model.params, model.act_rng, data))
    print(hk.experimental.tabulate(model.raw_action)(
        model.params, model.act_rng, data))
