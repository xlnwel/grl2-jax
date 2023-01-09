import os
import math
import logging
import jax
from jax import lax, nn, random
import jax.numpy as jnp
import haiku as hk

from core.log import do_logging
from core.elements.model import Model as ModelBase
from core.typing import AttrDict, dict2AttrDict
from tools.file import source_file
from jax_tools import jax_dist
from tools.display import print_dict_info

# register ppo-related networks 
source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))
logger = logging.getLogger(__name__)


def construct_fake_data(env_stats, aid, batch_size=1):
    basic_shape = (batch_size, 1, len(env_stats.aid2uids[aid]))
    shapes = env_stats.obs_shape[aid]
    dtypes = env_stats.obs_dtype[aid]
    action_dim = env_stats.action_dim[aid]
    data = {k: jnp.zeros((*basic_shape, *v), dtypes[k]) 
        for k, v in shapes.items()}
    data = dict2AttrDict(data)
    data.setdefault('global_state', data.obs)
    data.setdefault('hidden_state', data.obs)
    data.action = jnp.zeros((*basic_shape, action_dim), jnp.float32)
    data.state_reset = jnp.zeros(basic_shape, jnp.float32)

    # print_dict_info(data)

    return data

class Model(ModelBase):
    def add_attributes(self):
        self.imaginary_params = dict2AttrDict({'imaginary': True})
        self.params.imaginary = False

        self._initial_state = None

    def build_nets(self):
        aid = self.config.get('aid', 0)
        self.is_action_discrete = self.env_stats.is_action_discrete[aid]
        data = construct_fake_data(self.env_stats, aid)

        self.params.policy, self.modules.policy = self.build_net(
            data.obs, data.state_reset, data.state, data.action_mask, name='policy')
        self.params.value, self.modules.value = self.build_net(
            data.global_state, data.state_reset, data.state, name='value')
        self.sync_imaginary_params()

    def compile_model(self):
        self.jit_action = jax.jit(self.raw_action, static_argnames=('evaluation'))

    @property
    def theta(self):
        return self.params

    def switch_params(self, imaginary):
        self.params, self.imaginary_params = self.imaginary_params, self.params
        self.check_params(imaginary)

    def check_params(self, imaginary):
        assert self.params.imaginary == imaginary, (self.params.imaginary, imaginary)
        assert self.imaginary_params.imaginary == 1-imaginary, (self.params.imaginary, imaginary)

    def sync_imaginary_params(self):
        for k, v in self.params.items():
            if k != 'imaginary':
                self.imaginary_params[k] = v

    def raw_action(
        self, 
        params, 
        rng, 
        data, 
        evaluation=False, 
    ):
        rngs = random.split(rng, 3)
        state = data.pop('state', AttrDict())
        data = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 1), data)
        act_out, policy_state = self.modules.policy(
            params.policy, 
            rngs[0], 
            data.obs, 
            data.state_reset, 
            state.policy, 
            action_mask=data.action_mask, 
        )
        state.policy = policy_state
        act_dist = self.policy_dist(act_out, evaluation)

        if self.is_action_discrete:
            stats = {'mu_logits': act_dist.logits}
        else:
            loc = act_dist.loc
            stats = {
                'mu_loc': loc,
                'mu_scale': act_dist.scale_diag, 
            }

        if evaluation:
            action = act_dist.mode()
            stats = {}
        else:
            action, logprob = act_dist.sample_and_log_prob(seed=rngs[1])
            value, value_state = self.modules.value(
                params.value, 
                rngs[2], 
                data.global_state, 
                data.state_reset, 
                state.value
            )
            state.value = value_state
            stats.update({'mu_logprob': logprob, 'value': value})
        action, stats = jax.tree_util.tree_map(
            lambda x: jnp.squeeze(x, 1), (action, stats))
        if state.policy is None and state.value is None:
            state = None
        
        return action, stats, state

    def compute_value(self, data):
        @jax.jit
        def comp_value(params, rng, global_state, state_reset=None, state=None):
            v, _ = self.modules.value(
                params.value, rng, 
                global_state, state_reset, state
            )
            return v
        self.act_rng, rng = random.split(self.act_rng)
        value = comp_value(self.params, rng, **data)
        return value

    def policy_dist(self, act_out, evaluation):
        if self.is_action_discrete:
            if evaluation and self.config.get('eval_act_temp', 0) > 0:
                act_out = act_out / self.config.eval_act_temp
            dist = jax_dist.Categorical(logits=act_out)
        else:
            loc, scale = act_out
            if evaluation and self.config.get('eval_act_temp', 0) > 0:
                scale = scale * self.config.eval_act_temp
            dist = jax_dist.MultivariateNormalDiag(loc, scale)

        return dist

    """ RNN Operators """
    def get_initial_state(self, batch_size):
        aid = self.config.get('aid', 0)
        data = construct_fake_data(self.env_stats, aid, batch_size)
        _, policy_state = self.modules.policy(
            self.params.policy, 
            self.act_rng, 
            data.obs, 
            data.state_reset
        )
        _, value_state = self.modules.value(
            self.params.value, 
            self.act_rng, 
            data.global_state, 
            data.state_reset
        )
        self._initial_state = AttrDict(
            policy=jax.tree_util.tree_map(jnp.zeros_like, policy_state), 
            value=jax.tree_util.tree_map(jnp.zeros_like, value_state), 
        )
        return self._initial_state
    
    @property
    def state_size(self):
        if self.config.policy.rnn_type is None and self.config.value.rnn_type is None:
            return None
        state_size = AttrDict(
            policy=self.config.policy.rnn_units, 
            value=self.config.value.rnn_units, 
        )
        return state_size
    
    @property
    def state_keys(self):
        if self.config.policy.rnn_type is None and self.config.value.rnn_type is None:
            return None
        key_map = {
            'lstm': hk.LSTMState._fields, 
            'gru': None, 
            None: None
        }
        state_keys = AttrDict(
            policy=key_map[self.config.policy.rnn_type], 
            value=key_map[self.config.value.rnn_type], 
        )
        return state_keys

    @property
    def state_type(self):
        if self.config.policy.rnn_type is None and self.config.value.rnn_type is None:
            return None
        type_map = {
            'lstm': hk.LSTMState, 
            'gru': None, 
            None: None
        }
        state_type = AttrDict(
            policy=type_map[self.config.policy.rnn_type], 
            value=type_map[self.config.value.rnn_type], 
        )
        return state_type


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


def create_model(
    config, 
    env_stats, 
    name='lka_v2', 
    **kwargs
): 
    config = setup_config_from_envstats(config, env_stats)

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
