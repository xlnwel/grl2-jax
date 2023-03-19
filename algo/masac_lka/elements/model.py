import os
import jax
from jax import random
import jax.numpy as jnp
import chex

from core.mixin.model import update_params
from core.typing import AttrDict
from tools.display import print_dict_info
from tools.file import source_file
from tools.utils import batch_dicts
from algo.lka_common.elements.model import *
from algo.masac.elements.utils import concat_along_unit_dim

# register ppo-related networks 
source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


LOOKAHEAD = 'lookahead'


class Model(LKAModelBase):
    def add_attributes(self):
        super().add_attributes()
        self.target_params = AttrDict()

    def build_nets(self):
        aid = self.config.get('aid', 0)
        data = construct_fake_data(self.env_stats, aid=aid)

        # policies for each agent
        self.params.policies = []
        policy_init, self.modules.policy = self.build_net(
            name='policy', return_init=True)
        self.rng, policy_rng, q_rng = random.split(self.rng, 3)
        self.act_rng = self.rng
        for rng in random.split(policy_rng, self.n_agents):
            self.params.policies.append(policy_init(
                rng, data.obs, data.state_reset, data.state, data.action_mask
            ))
            self.params.policies[-1][LOOKAHEAD] = False
        
        self.params.Qs = []
        q_init, self.modules.Q = self.build_net(
            name='Q', return_init=True)
        global_state = data.global_state[:, :, :1]
        for rng in random.split(q_rng, self.config.n_Qs):
            self.params.Qs.append(q_init(
                rng, global_state, data.joint_action, data.state_reset, data.state
            ))
        self.params.temp, self.modules.temp = self.build_net(name='temp')

        self.sync_target_params()
        self.sync_lookahead_params()

    def compile_model(self):
        self.jit_action = jax.jit(self.raw_action, static_argnames=('evaluation'))
        self.jit_forward_policy = jax.jit(
            self.forward_policy, static_argnames=('return_state', 'evaluation'))

    @property
    def target_theta(self):
        return self.target_params

    def sync_target_params(self):
        self.target_params = self.params.copy()
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
        rngs = random.split(rng, self.n_agents)
        all_actions = []
        all_stats = []
        all_states = []
        for aid, (p, rng) in enumerate(zip(params.policies, rngs)):
            agent_rngs = random.split(rng, 2)
            d = data[aid]
            if self.has_rnn:
                state = d.pop('state', AttrDict())
                d = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 1) , d)
                d.state = state
            else:
                state = AttrDict()
            act_out, state = self.forward_policy(p, agent_rngs[0], d)
            act_dist = self.policy_dist(act_out, evaluation)

            if evaluation:
                action = act_dist.sample(seed=agent_rngs[1])
                stats = AttrDict()
            else:
                action = act_dist.sample(seed=agent_rngs[1])
                stats = act_dist.get_stats('mu')
            if not self.is_action_discrete:
                action = jnp.tanh(action)
            if self.has_rnn:
                action, stats = jax.tree_util.tree_map(
                    lambda x: jnp.squeeze(x, 1), (action, stats))
                all_states.append(state)
            else:
                all_states = None

            all_actions.append(action)
            all_stats.append(stats)

        action = concat_along_unit_dim(all_actions)
        stats = batch_dicts(all_stats, func=concat_along_unit_dim)
        if self.has_rnn:
            all_states = batch_dicts(all_states, func=concat_along_unit_dim)

        return action, stats, all_states


    """ RNN Operators """
    def get_initial_state(self, batch_size):
        if self._initial_state is not None:
            return self._initial_state
        if not self.has_rnn:
            return None
        data = construct_fake_data(self.env_stats, batch_size)
        states = []
        for p, q, d in zip(self.params.policies, self.params.qs, data):
            state = AttrDict()
            _, state.policy = self.modules.policy(
                p, 
                self.act_rng, 
                d.obs, 
            )
            _, state.q = self.modules.Q(
                q, 
                self.act_rng, 
                d.global_state, 
            )
            states.append(state)
        self._initial_state = jax.tree_util.tree_map(jnp.zeros_like, states)

        return self._initial_state


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


# if __name__ == '__main__':
#     from tools.yaml_op import load_config
#     from env.func import create_env
#     from tools.display import pwc
#     config = load_config('algo/zero_mr/configs/magw_a2c')
    
#     env = create_env(config.env)
#     model = create_model(config.model, env.stats())
#     data = construct_fake_data(env.stats(), 0)
#     print(model.action(model.params, data))
#     pwc(hk.experimental.tabulate(model.raw_action)(model.params, data), color='yellow')
