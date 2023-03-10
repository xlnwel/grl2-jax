import os
import logging
import numpy as np
import jax
from jax import lax, nn, random
import jax.numpy as jnp
import haiku as hk
import chex

from core.log import do_logging
from core.elements.model import Model as ModelBase
from core.mixin.model import update_params
from core.typing import AttrDict, dict2AttrDict
from jax_tools import jax_dist, jax_utils
from tools.file import source_file
from tools.utils import batch_dicts
from tools.display import print_dict_info
from tools.timer import Timer
from algo.masac.elements.utils import concat_along_unit_dim

# register ppo-related networks 
source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))
logger = logging.getLogger(__name__)


LOOKAHEAD = 'lookahead'


def construct_fake_data(env_stats, aid, batch_size=1):
    n_units = env_stats.n_units
    basic_shape = (batch_size, 1, n_units)
    shapes = env_stats.obs_shape[aid]
    dtypes = env_stats.obs_dtype[aid]
    action_dim = env_stats.action_dim[aid]
    data = {k: jnp.zeros((*basic_shape, *v), dtypes[k]) 
        for k, v in shapes.items()}
    data = dict2AttrDict(data)
    data.setdefault('global_state', data.obs)
    data.setdefault('hidden_state', data.obs)
    data.action = jnp.zeros((*basic_shape, action_dim), jnp.float32)
    data.joint_action = jnp.zeros((*basic_shape, action_dim), jnp.float32)
    data.state_reset = jnp.zeros(basic_shape, jnp.float32)

    # print_dict_info(data)

    return data


class Model(ModelBase):
    def add_attributes(self):
        self.aid2uids = self.env_stats.aid2uids
        self.lookahead_params = [{LOOKAHEAD: True} for _ in self.aid2uids]
        self.target_params = AttrDict()

        self._initial_state = None

    def build_nets(self):
        aid = self.config.get('aid', 0)
        self.is_action_discrete = self.env_stats.is_action_discrete[aid]
        data = construct_fake_data(self.env_stats, aid)

        # policies for each agent
        self.params.policies = []
        policy_init, self.modules.policy = self.build_net(
            name='policy', return_init=True)
        self.rng, policy_rng, q_rng = random.split(self.rng, 3)
        self.act_rng = self.rng
        for rng in random.split(policy_rng, self.env_stats.n_agents):
            self.params.policies.append(policy_init(
                rng, data.obs, data.state_reset, data.state, data.action_mask
            ))
            self.params.policies[-1][LOOKAHEAD] = False
        
        self.params.Qs = []
        q_init, self.modules.Q = self.build_net(
            name='Q', return_init=True)
        global_state = data.global_state[:, :, :1]
        action = data.action.reshape(*data.action.shape[:2], 1, -1)
        for rng in random.split(q_rng, self.config.n_Qs):
            self.params.Qs.append(q_init(
                rng, global_state, action, data.state_reset, data.state
            ))
        self.params.temp, self.modules.temp = self.build_net(name='temp')
        
        self.sync_target_params()
        self.sync_lookahead_params()

    def compile_model(self):
        self.jit_action = jax.jit(self.raw_action, static_argnames=('evaluation'))

    @property
    def theta(self):
        return self.params
    
    @property
    def target_theta(self):
        return self.target_params

    def switch_params(self, lookahead, aids=None):
        if aids is None:
            aids = np.arange(len(self.aid2uids))
        for i in aids:
            self.params.policies[i], self.lookahead_params[i] = \
                self.lookahead_params[i], self.params.policies[i]
        self.check_params(lookahead, aids)

    def check_params(self, lookahead, aids=None):
        if aids is None:
            aids = np.arange(len(self.aid2uids))
        for i in aids:
            assert self.params.policies[i][LOOKAHEAD] == lookahead, (self.params.policies[i][LOOKAHEAD], lookahead)
            assert self.lookahead_params[i][LOOKAHEAD] == 1-lookahead, (self.lookahead_params[i][LOOKAHEAD], lookahead)

    def sync_lookahead_params(self):
        self.lookahead_params = [p.copy() for p in self.params.policies]
        for p in self.lookahead_params:
            p[LOOKAHEAD] = True
    
    def sync_target_params(self):
        self.target_params = self.params
        chex.assert_trees_all_close(self.params, self.target_params)

    def update_target_params(self):
        self.target_params = update_params(
            self.params, self.target_params, self.config.polyak)

    def raw_action(
        self, 
        params, 
        rng, 
        data, 
        evaluation=False, 
    ):
        rngs = random.split(rng, len(self.aid2uids))
        all_actions = []
        all_stats = []
        all_states = []
        for aid, (p, rng) in enumerate(zip(params.policies, rngs)):
            d = data[aid]
            state = d.pop('state', AttrDict())
            d = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 1), d)
            act_out, state.policy = self.modules.policy(
                p, 
                rngs[0], 
                d.obs, 
                d.state_reset, 
                state.policy, 
                action_mask=d.action_mask, 
            )
            act_dist = self.policy_dist(act_out, evaluation)

            if evaluation:
                action = act_dist.mode()
                stats = AttrDict()
            else:
                action = act_dist.sample(seed=rngs[1])
                stats = act_dist.get_stats('mu')
            action, stats = jax.tree_util.tree_map(
                lambda x: jnp.squeeze(x, 1), (action, stats))
            if not self.is_action_discrete:
                action = jnp.tanh(action)
            if state.policy is None and state.value is None:
                state = None

            all_actions.append(action)
            all_stats.append(stats)
            all_states.append(state)

        action = concat_along_unit_dim(all_actions)
        stats = batch_dicts(all_stats, func=concat_along_unit_dim)
        if state is None:
            states = None
        else:
            states = batch_dicts(all_states, func=concat_along_unit_dim)

        return action, stats, states
        
    def policy_dist(self, act_out, evaluation=False):
        if self.is_action_discrete:
            if evaluation and self.config.get('eval_act_temp', 0) > 0:
                act_out = act_out / self.config.eval_act_temp
            dist = jax_dist.Categorical(logits=act_out)
        else:
            loc, scale = act_out
            if evaluation and self.config.get('eval_act_temp', 0) > 0:
                scale = scale * self.config.eval_act_temp
            dist = jax_dist.MultivariateNormalDiag(
                loc, scale, joint_log_prob=self.config.joint_log_prob)

        return dist

    """ RNN Operators """
    # TODO
    def get_initial_state(self, batch_size):
        aid = self.config.get('aid', 0)
        data = construct_fake_data(self.env_stats, aid, batch_size)
        _, policy_state = self.modules.policy(
            self.params.policy, 
            self.act_rng, 
            data.obs, 
            data.state_reset
        )
        qs_state = []
        for q_params in self.params.Qs:
            _, q_state = self.modules.Q(
                q_params, 
                self.act_rng, 
                data.global_state, 
                data.action, 
                data.state_reset
            )
            qs_state.append(q_state)
        if all([s is None for s in qs_state]):
            qs_state = None
        self._initial_state = AttrDict(
            policy=jax.tree_util.tree_map(jnp.zeros_like, policy_state), 
            qs=jax.tree_util.tree_map(jnp.zeros_like, qs_state), 
        )
        return self._initial_state
    
    @property
    def state_size(self):
        if self.config.policy.rnn_type is None and self.config.Q.rnn_type is None:
            return None
        state_size = AttrDict(
            policy=self.config.policy.rnn_units, 
            qs=self.config.Q.rnn_units, 
        )
        return state_size
    
    @property
    def state_keys(self):
        if self.config.policy.rnn_type is None and self.config.Q.rnn_type is None:
            return None
        key_map = {
            'lstm': hk.LSTMState._fields, 
            'gru': None, 
            None: None
        }
        state_keys = AttrDict(
            policy=key_map[self.config.policy.rnn_type], 
            qs=key_map[self.config.Q.rnn_type], 
        )
        return state_keys

    @property
    def state_type(self):
        if self.config.policy.rnn_type is None and self.config.Q.rnn_type is None:
            return None
        type_map = {
            'lstm': hk.LSTMState, 
            'gru': None, 
            None: None
        }
        state_type = AttrDict(
            policy=type_map[self.config.policy.rnn_type], 
            qs=type_map[self.config.Q.rnn_type], 
        )
        return state_type


def setup_config_from_envstats(config, env_stats):
    if 'aid' in config:
        aid = config['aid']
        config.policy.action_dim = env_stats.action_dim[aid]
        config.policy.is_action_discrete = env_stats.is_action_discrete[aid]
        config.Q.is_action_discrete = env_stats.is_action_discrete[aid]
    else:
        config.policy.action_dim = env_stats.action_dim
        config.policy.is_action_discrete = env_stats.is_action_discrete
        config.Q.is_action_discrete = env_stats.is_action_discrete[aid]

    return config


def create_model(
    config, 
    env_stats, 
    name='masac', 
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
