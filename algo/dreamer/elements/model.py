import os
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
from jax_tools import jax_dist
from env.typing import EnvOutput
from nn.rssm import RSSMState


source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))
logger = logging.getLogger(__name__)


def construct_fake_data(env_stats, config, aid, batch_size=1):
    basic_shape = (batch_size, 1, len(env_stats.aid2uids[aid]))
    basic_shape2 = (batch_size, len(env_stats.aid2uids[aid]))
    shapes = env_stats.obs_shape[aid]
    dtypes = env_stats.obs_dtype[aid]
    action_dim = env_stats.action_dim[aid]

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
    def add_attributes(self):
        self.lookahead_params = dict2AttrDict({'lookahead': True})
        self.params.lookahead = False
        
        self._initial_state, self._state_initial_rssm, self._obs_initial_rssm = None, None, None

    def build_nets(self):
        aid = self.config.get('aid', 0)
        self.is_action_discrete = self.env_stats.is_action_discrete[aid]
        data = construct_fake_data(self.env_stats, self.config, aid)

        self.params.rssm_embed, self.modules.rssm_embed = self.build_net(
            data.obs_za, name='embedlayer')
        self.params.rssm_rnn, self.modules.rssm_rnn = self.build_net(
            data.obs_rssm_embed, data.state_reset2, data.obs_deter, name='rssmrnnlayer')
        self.params.rssm_trans, self.modules.rssm_trans = self.build_net(
            data.obs_deter, name='translayer')
        self.params.rssm_repre, self.modules.rssm_repre = self.build_net(
            data.obs_hx, name='reprelayer')
        self.rssm = RSSM(
            embed_layer=self.modules.rssm_embed,
            rssm_rnn_layer=self.modules.rssm_rnn,
            trans_layer=self.modules.rssm_trans,
            repre_layer=self.modules.rssm_repre
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
        self.params.policy, self.modules.policy = self.build_net(
            data.obs_rssm, data.state_reset, data.state, data.action_mask, name='policy')
        self.params.value, self.modules.value = self.build_net(
            data.state_rssm, data.state_reset, data.state, name='value')
        self.sync_lookahead_params()
    
    def compile_model(self):
        self.jit_action = jax.jit(self.raw_action, static_argnames=('evaluation'))
            
    @property
    def theta(self):
        return self.params

    def switch_params(self, lookahead):
        self.params.policy, self.lookahead_params.policy = \
            self.lookahead_params.policy, self.params.policy
        self.params.value, self.lookahead_params.value = \
            self.lookahead_params.value, self.params.value

    def sync_lookahead_params(self):
        for k, v in self.params.items():
            # TODO: We only consider copy the parameters of `value' and `policy' network.
            if k in ['value', 'policy']:
                self.lookahead_params[k] = v

    def build_attribute(self, *args, name, attribute_name, **kwargs):
        def build(*args, **kwargs):
            net = create_network(self.config[name], name)
            return getattr(net, attribute_name)(*args, **kwargs)
        func = hk.transform(build)
        return func.apply

    def raw_action(
        self,
        params,
        rng,
        data,
        evaluation=False,
    ):
        rngs = random.split(rng, 9)
        state_rssm = data.pop('state_rssm', AttrDict())
        obs_rssm = data.pop('obs_rssm', AttrDict())
        state = data.pop('state', AttrDict())
        data = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 1), data)
        # obtain the rssm state information of observation
        obs_embed = self.modules.obs_encoder(params.obs_encoder, rngs[0], data.obs)
        obs_post, _ = self.modules.rssm(params.rssm, rngs[1], data.prev_action, obs_embed, data.state_reset, obs_rssm, imagine=False, just_step=True)
        obs_rssm_feat = jnp.expand_dims(self.modules.rssm_feat(params.rssm, rngs[2], obs_post), 1)
        # obtain the rssm state information of state
        state_embed = self.modules.state_encoder(params.state_encoder, rngs[3], data.global_state)
        state_post, _ = self.modules.rssm(params.rssm, rngs[4], data.prev_action, state_embed, data.state_reset, state_rssm, imagine=False, just_step=True)
        state_rssm_feat = jnp.expand_dims(self.modules.rssm_feat(params.rssm, rngs[5], state_post), 1)
        act_out, state.policy = self.modules.policy(
            params.policy,
            rngs[6],
            obs_rssm_feat,
            data.state_reset,
            state.policy,
            action_mask=data.action_mask,
        )
        act_dist = self.policy_dist(act_out, evaluation)

        if evaluation:
            action = act_dist.mode()
            stats = {}
        else:
            stats = act_dist.get_stats('mu')
            action, logprob = act_dist.sample_and_log_prob(seed=rngs[7])
            value, state.value = self.modules.value(
                params.value, 
                rngs[8],
                state_rssm_feat,
                data.state_reset, 
                state.value
            )
            stats.update({'mu_logprob': logprob, 'value': value})
        action, stats = jax.tree_util.tree_map(
            lambda x: jnp.squeeze(x, 1), (action, stats))
        if state.policy is None and state.value is None:
            state = None

        # TODO 
        return action, stats, state, state_post, obs_post

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
            state_embed = self.modules.state_encoder(params.state_encoder, rng, global_state)
            state_post, _ = self.modules.rssm(params.rssm, rng, action, state_embed, state_reset, state_rssm, imagine=False, just_step=True)
            state_rssm_feat = jnp.expand_dims(self.modules.rssm_feat(params.rssm, rng, state_post), 1)
            v, _ = self.modules.value(
                params.value, rng,
                state_rssm_feat, state_reset, state
            )
            return v
        self.act_rng, rng = random.split(self.act_rng)
        value = comp_value(self.params, rng, **data)
        return value
        
    """ Define initial state for RNN practice. """
    def get_initial_state(self, batch_size):
        aid = self.config.get('aid', 0)
        data = construct_fake_data(self.env_stats, self.config, aid, batch_size)
        state_embed = self.modules.state_encoder(self.params.state_encoder, self.act_rng, data.global_state)
        rssm = self.modules.rssm(self.params.rssm, self.act_rng, data.action, state_embed, data.state_reset, imagine=True, just_step=True)
        rssm_feat = self.modules.rssm_feat(self.params.rssm, self.act_rng, rssm)
        rssm_feat = jnp.expand_dims(rssm_feat, 1) # [B, U, *] -> [B, T, U, *]
        # obtain rssm initial state
        self._state_initial_rssm = jax.tree_util.tree_map(jnp.zeros_like, rssm)
        self._obs_initial_rssm = jax.tree_util.tree_map(jnp.zeros_like, rssm)

        # TODO
        _, policy_state = self.modules.policy(
            self.params.policy, 
            self.act_rng, 
            rssm_feat,
            data.state_reset
        )
        _, value_state = self.modules.value(
            self.params.value, 
            self.act_rng, 
            rssm_feat,
            data.state_reset
        )
        self._initial_state = AttrDict(
            policy=jax.tree_util.tree_map(jnp.zeros_like, policy_state), 
            value=jax.tree_util.tree_map(jnp.zeros_like, value_state), 
        )

        return self._initial_state, self._state_initial_rssm, self._obs_initial_rssm

    def process_action(self, action):
        action = nn.one_hot(action, self.env_stats.action_dim[0])
        return action

    # def next_hidden(self, params, rng, obs, action, stats):
    #     """
    #         According to the current hidden and action, then we obtain the hidden of the next timestep.
    #     """
    #     raise NotImplementedError

    # def reward(self, params, rng):
    #     """
    #         Get the prediction of the reward function.
    #     """
    #     raise NotImplementedError


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
    # print(model.action(model.params, data))
    # pwc(hk.experimental.tabulate(model.raw_action)(model.params, data), color='yellow')
