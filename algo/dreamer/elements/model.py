import os
from re import S
import numpy as np
import logging
import collections
import jax
from jax import lax, nn, random
import jax.numpy as jnp
import haiku as hk

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

from core.elements.model import Model as ModelBase
from core.typing import dict2AttrDict, AttrDict
from nn.rssm import RSSM
from nn.func import create_network
from tools.file import source_file
from tools.utils import batch_dicts
from jax_tools import jax_dist
from env.typing import EnvOutput
from nn.rssm import RSSMState
from algo.dreamer.run import concat_along_unit_dim


source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))
logger = logging.getLogger(__name__)


LOOKAHEAD = 'lookahead'

def concate_along_time_dim(x):
    x = jnp.concatenate(x, axis=1)
    return x

def concate_along_time_dim(x):
    x = jnp.concatenate(x, axis=1)
    return x

def construct_fake_data(env_stats, config, n_units, batch_size=1):
    basic_shape = (batch_size, 1, n_units)
    basic_shape2 = (batch_size, n_units)
    shapes = env_stats.obs_shape[0]
    dtypes = env_stats.obs_dtype[0]
    action_dim = env_stats.action_dim[0]

    data = {k: jnp.zeros((*basic_shape, *v), dtypes[k]) 
        for k, v in shapes.items()}
    
    data.update({
        'obs_embed': jnp.zeros((*basic_shape2, 1)),
        'obs_za': jnp.zeros((*basic_shape2, config.stoch_size + action_dim)),
        'obs_rssm_embed': jnp.zeros((*basic_shape2, config.deter_size)),
        'obs_deter': jnp.zeros((*basic_shape2, config.deter_size)),
        'obs_hx': jnp.zeros((*basic_shape2, config.deter_size + 1)),
        'obs_rssm': jnp.zeros((*basic_shape, config.stoch_size + config.deter_size)),
        'state_rssm': jnp.zeros((*basic_shape, config.stoch_size + config.deter_size)),
        'state_reset': jnp.zeros(basic_shape),
        'state_reset2': jnp.zeros(basic_shape2)
    })
    rssm_state = RSSMState(
        mean=jnp.zeros((*basic_shape2, config.stoch_size)),
        std=jnp.zeros((*basic_shape2, config.stoch_size)),
        stoch=jnp.zeros((*basic_shape2, config.stoch_size)),
        deter=jnp.zeros((*basic_shape2, config.deter_size)),
    )
    data.update({'rssm_state': rssm_state})

    data = dict2AttrDict(data)
    data.action = jnp.zeros((*basic_shape, action_dim), jnp.float32)
    return data


class Model(ModelBase):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.act_rng, self.model_rng = random.split(self.rng, 2)
    
    def model_rollout(self, state, rollout_length):
        self.model_rng, model_rng = random.split(self.model_rng, 2)
        return self.jit_model_rollout(
            self.params, model_rng, state, rollout_length
        )

    def add_attributes(self):
        self.aid2uids = self.env_stats.aid2uids
        self.n_units = self.env_stats.n_units
        self.lookahead_params = AttrDict()
        self._initial_state = None

    def build_nets(self):
        aid = self.config.get('aid', 0)
        self.is_action_discrete = self.env_stats.is_action_discrete[aid]
        data = construct_fake_data(self.env_stats, self.config, self.n_units)

        self.params.rssm_embed, self.modules.rssm_embed = self.build_net(
            data.obs_za, name='embedlayer')
        self.params.rssm_rnn, self.modules.rssm_rnn = self.build_net(
            data.obs_rssm_embed, data.state_reset2, data.obs_deter, name='rssmrnnlayer')
        self.params.rssm_trans, self.modules.rssm_trans = self.build_net(
            data.obs_deter, name='translayer')
        self.params.rssm_repre, self.modules.rssm_repre = self.build_net(
            data.obs_hx, name='reprelayer')
        self.rng, rssm_rng, policy_rng, value_rng = random.split(self.rng, 4)
        self.act_rng = self.rng
        self.rssm = RSSM(
            embed_layer=self.modules.rssm_embed,
            rssm_rnn_layer=self.modules.rssm_rnn,
            trans_layer=self.modules.rssm_trans,
            repre_layer=self.modules.rssm_repre,
            stoch=self.config.stoch_size,
            deter=self.config.deter_size,
            rnn_type=self.config.rssm_rnn_type,
            rng=rssm_rng,
        )
        self.params.reward, self.modules.reward = self.build_net(
            data.state_rssm, data.action, name='reward')
        self.params.discount, self.modules.discount = self.build_net(
            data.state_rssm, name='discount')
        self.params.state_encoder, self.modules.state_encoder = self.build_net(
            data.global_state, name='stateencoder')
        self.params.obs_encoder, self.modules.obs_encoder = self.build_net(
            data.obs, name='obsencoder')
        self.params.decoder, self.modules.decoder = self.build_net(
            data.state_rssm, name='decoder')
        
        self.params.policies, self.params.vs = [], []
        policy_init, self.modules.policy = self.build_net(
            name='policy', return_init=True
        )
        for rng in random.split(policy_rng, self.env_stats.n_agents):
            self.params.policies.append(policy_init(
                rng, data.obs_rssm, data.state_reset, data.state, data.action_mask))
            self.params.policies[-1][LOOKAHEAD] = False
        value_init, self.modules.value = self.build_net(
            name='value', return_init=True
        )
        for rng in random.split(value_rng, self.env_stats.n_agents):
            self.params.vs.append(value_init(
                rng, data.state_rssm, data.state_reset, data.state))
        self.sync_lookahead_params()

    def compile_model(self):
        self.jit_action = jax.jit(self.raw_action, static_argnames=('evaluation'))
        self.jit_model_rollout = jax.jit(self.raw_model_rollout, static_argnames=('rollout_length'))
            
    @property
    def theta(self):
        return self.params

    def switch_params(self, lookahead, aids=None):
        if aids is None:
            aids = np.arange(self.n_units)
        for i in aids:
            self.params.policies[i], self.lookahead_params.policies[i] = \
                self.lookahead_params.policies[i], self.params.policies[i]
            self.params.vs[i], self.lookahead_params.vs[i] = \
                self.lookahead_params.vs[i], self.params.vs[i]
        self.check_params(lookahead, aids)

    def check_params(self, lookahead, aids=None):
        if aids is None:
            aids = np.arange(self.n_units)
        for i in aids:
            assert self.params.policies[i][LOOKAHEAD] == lookahead, (self.params.policies[i][LOOKAHEAD], lookahead)
            assert self.lookahead_params.policies[i][LOOKAHEAD] == 1-lookahead, (self.lookahead_params.policies[i][LOOKAHEAD], lookahead)

    def sync_lookahead_params(self):
        self.lookahead_params.policies = [p.copy() for p in self.params.policies]
        for p in self.lookahead_params.policies:
            p[LOOKAHEAD] = True
        self.lookahead_params.vs = [v.copy() for v in self.params.vs]

    def build_attribute(self, *args, name, attribute_name, **kwargs):
        def build(*args, **kwargs):
            net = create_network(self.config[name], name)
            return getattr(net, attribute_name)(*args, **kwargs)
        func = hk.transform(build)
        return func.apply

    def get_policy_feat(self, params, rng, data, obs_rssm_state=None):
        rngs = random.split(rng, 2)
        obs_embed = self.modules.obs_encoder(params.obs_encoder, rngs[0], data.obs)
        obs_post, _ = self.rssm.obs_step(params, rngs[1], obs_rssm_state, data.prev_action, obs_embed, data.state_reset)
        return self.rssm.get_feat(obs_post)
    
    def get_value_feat(self, params, rng, data, state_rssm_state=None):
        rngs = random.split(rng, 2)
        state_embed = self.module.state_encoder(params.state_encoder, rngs[0], data.state)
        state_post, _ = self.rssm.obs_step(params, rngs[1], state_rssm_state, data.prev_action, state_embed, data.state_reset)
        return self.rssm.get_feat(state_post)

    def raw_model_rollout(self, params, rng, state, rollout_length=10):
        tot_actions, tot_rewards, tot_discount = [], [], []
        tot_state_rssm, tot_obs_rssm = [], []
        tot_next_state_rssm, tot_next_obs_rssm = [], []
        tot_state, tot_state_reset = [], []
        tot_stats = []
        for _ in range(rollout_length):
            rng, act_rng, img_rng, reward_rng, discount_rng = random.split(rng, 5)
            act_rngs = random.split(act_rng, self.n_units)
            
            state_rssm_state = state.state_rssm_state
            obs_rssm_state = state.obs_rssm_state
            state_rssm_feat = jnp.expand_dims(self.rssm.get_feat(state_rssm_state), 1)
            obs_rssm_feat = jnp.expand_dims(self.rssm.get_feat(obs_rssm_state), 1)
            state_reset = jnp.zeros((state_rssm_state.stoch.shape[0], 1, self.n_units))

            all_actions, all_stats, all_states = [], [], []
            for aid, (p, act_rng) in enumerate(zip(params.policies, act_rngs)):
                arngs = random.split(act_rng, 3)
                obs_feat = obs_rssm_feat[..., aid:aid+1, :]
                state_feat = state_rssm_feat[..., aid:aid+1, :]
                astate_policy = jax.tree_util.tree_map(lambda x: x[..., aid:aid+1, :], state.policy)
                astate_value = jax.tree_util.tree_map(lambda x: x[..., aid:aid+1, :], state.value)
                astate_reset = jax.tree_util.tree_map(lambda x: x[..., aid:aid+1], state_reset)

                astate = AttrDict()
                act_out, astate.policy = self.modules.policy(
                    p,
                    arngs[0],
                    obs_feat,
                    astate_reset,
                    astate_policy,
                    action_mask=None,
                )
                act_dist = self.policy_dist(act_out, False)
                
                stats = act_dist.get_stats('mu')
                action, logprob = act_dist.sample_and_log_prob(seed=arngs[1])
                
                value, astate.value = self.modules.value(
                    params.vs[aid],
                    arngs[2],
                    state_feat,
                    astate_reset,
                    astate_value,
                )
                stats.update({'mu_logprob': logprob, 'value': value})
                
                action, stats = jax.tree_util.tree_map(
                    lambda x: jnp.squeeze(x, 1), (action, stats))
                if not self.is_action_discrete:
                    action = jnp.tanh(action)
                if astate.policy is None:
                    astate = None
                
                all_actions.append(action)
                all_stats.append(stats)
                all_states.append(astate)

            action = concat_along_unit_dim(all_actions)
            stats = batch_dicts(all_stats, func=concat_along_unit_dim)
            if astate is None:
                state = None
            else:
                state = batch_dicts(all_states, func=concat_along_unit_dim)
            
            state_reset = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, 1), state_reset)
            onehot_action = self.process_action(action)
            next_state_rssm_state = self.rssm.img_step(params, img_rng, state_rssm_state, onehot_action, state_reset)
            next_obs_rssm_state = self.rssm.img_step(params, img_rng, obs_rssm_state, onehot_action, state_reset)
            state.state_rssm_state = next_state_rssm_state
            state.obs_rssm_state = next_obs_rssm_state
            
            next_state_rssm_feat = self.rssm.get_feat(next_state_rssm_state)
            next_obs_rssm_feat = self.rssm.get_feat(next_obs_rssm_state)

            reward_dist = self.modules.reward(params.reward, reward_rng, state_rssm_feat, jnp.expand_dims(onehot_action, 1))
            reward = reward_dist.mode()
            discount_dist = self.modules.discount(params.discount, discount_rng, next_state_rssm_feat)
            discount = discount_dist.mode()

            tot_actions.append(jnp.expand_dims(action, 1))
            tot_rewards.append(reward)
            tot_discount.append(jnp.expand_dims(discount, 1))
            tot_state_rssm.append(state_rssm_feat)
            tot_obs_rssm.append(obs_rssm_feat)
            tot_next_state_rssm.append(jnp.expand_dims(next_state_rssm_feat, 1))
            tot_next_obs_rssm.append(jnp.expand_dims(next_obs_rssm_feat, 1))
            tot_state_reset.append(jnp.expand_dims(state_reset, 1))
            tot_state.append(jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 1), state))
            tot_stats.append(jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 1), stats))

        tot_state_reset.append(jnp.expand_dims(state_reset, 1))
        tot_stats = batch_dicts(tot_stats, func=concate_along_time_dim)
        rollout_data = {
            'action': concate_along_time_dim(tot_actions),
            'reward': concate_along_time_dim(tot_rewards),
            'discount': concate_along_time_dim(tot_discount),
            'state_rssm': concate_along_time_dim(tot_state_rssm),
            'obs_rssm': concate_along_time_dim(tot_obs_rssm),
            'next_state_rssm': concate_along_time_dim(tot_next_state_rssm),
            'next_obs_rssm': concate_along_time_dim(tot_next_obs_rssm),
            'state_reset': concate_along_time_dim(tot_state_reset),
            'state': batch_dicts(tot_state, func=concate_along_time_dim),
        }
        rollout_data.update(tot_stats)
        
        rollout_data = dict2AttrDict(rollout_data)
        return rollout_data

    def raw_action(
        self,
        params,
        rng,
        data,
        evaluation=False,
    ):
        rngs = random.split(rng, self.n_units)
        all_actions = []
        all_stats = []
        all_states = []
        for aid, (p, rng) in enumerate(zip(params.policies, rngs)):
            d = data[aid]
            state = d.pop('state', AttrDict())
            d = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 1), d)
            
            arngs = random.split(rngs[aid], 3)
            obs_rssm_feat = jnp.expand_dims(self.get_policy_feat(params, arngs[0], d, state.obs_rssm_state), 1)
            act_out, state.policy = self.modules.policy(
                p, 
                arngs[1],
                obs_rssm_feat,
                d.state_reset, 
                state.policy, 
                action_mask=d.action_mask, 
            )
            act_dist = self.policy_dist(act_out, evaluation)

            if evaluation:
                action = act_dist.mode()
                stats = AttrDict()
            else:
                action = act_dist.sample(seed=arngs[2])
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

    def compute_value(self, data):
        @jax.jit
        def comp_value(params, rng, global_state, action, state_reset=None, state=None, state_rssm=None, obs_rssm=None):
            state_rssm_state = data.state.state_rssm_state
            state_rssm_feat = self.get_value_feat(params, rng, data, state_rssm_state)
            v, _ = self.modules.value(
                params.value, rng,
                state_rssm_feat, state_reset, state
            )
            return v
        self.act_rng, rng = random.split(self.act_rng)
        value = comp_value(self.params, rng, data)
        return value
        
    """ Define initial state for RNN practice. """
    def get_initial_state(self, batch_size):
        if self._initial_state is not None:
            return self._initial_state
        else:
            data = construct_fake_data(self.env_stats, self.config, self.n_units, batch_size)
            rssm = self.rssm.initial_rssm_state(self.params, self.act_rng, batch_size, self.n_units)
            rssm_feat = jnp.expand_dims(self.rssm.get_feat(rssm), 1) # [B, U, *] -> [B, T, U, *]

            _, policy_state = self.modules.policy(
                self.params.policies[0], 
                self.act_rng, 
                rssm_feat,
                data.state_reset
            )
            _, value_state = self.modules.value(
                self.params.vs[0], 
                self.act_rng, 
                rssm_feat,
                data.state_reset
            )
            self._initial_state = AttrDict(
                policy=jax.tree_util.tree_map(jnp.zeros_like, policy_state), 
                value=jax.tree_util.tree_map(jnp.zeros_like, value_state), 
                obs_rssm_state=jax.tree_util.tree_map(jnp.zeros_like, rssm),
                state_rssm_state=jax.tree_util.tree_map(jnp.zeros_like, rssm),
            )
            return self._initial_state

    def process_action(self, action):
        action = nn.one_hot(action, self.env_stats.action_dim[0])
        return action


def setup_config_from_envstats(config, env_stats):
    if 'aid' in config:
        aid = config['aid']
        config.decoder.out_size = env_stats.obs_shape[aid]['global_state'][0]
        config.policy.action_dim = env_stats.action_dim[aid]
        config.policy.is_action_discrete = env_stats.is_action_discrete[aid]
    else:
        config.decoder.out_size = env_stats.obs_shape['global_state'][0]
        config.policy.action_dim = env_stats.action_dim
        config.policy.is_action_discrete = env_stats.is_action_discrete
        config.policy.action_low = env_stats.action_low
        config.policy.action_high = env_stats.action_high

    return config


def create_model(
    config, 
    env_stats, 
    name='model', 
    **kwargs
): 
    config = setup_config_from_envstats(config, env_stats)

    return Model(
        config=config, 
        env_stats=env_stats,
        name=name,
        **kwargs)


if __name__ == '__main__':
    from tools.yaml_op import load_config
    from env.func import create_env
    # from tools.display import pwc
    # config = load_config('algo/zero_mr/configs/magw_a2c')
    project_direc = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    config_path =  os.path.join(project_direc, 'algo/dreamer/configs/escalation.yaml')
    config = load_config(config_path)
    config.model.stoch_size = config.model.deter_size = 32

    env = create_env(config.env)
    env_stats = env.stats()
    env_stats["aid2uids"] = [np.array([0 for _ in range(len(env_stats.aid2uids))])]
    model = create_model(config.model, env_stats)
    data = construct_fake_data(env.stats(), config.model, 0)
    print(data)
