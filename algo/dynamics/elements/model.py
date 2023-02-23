import os
import numpy as np
import logging
import collections
import jax
from jax import nn, random
import jax.numpy as jnp

from core.elements.model import Model as ModelBase
from core.typing import dict2AttrDict, AttrDict
from tools.file import source_file
from jax_tools import jax_dist
from env.typing import EnvOutput

# register ppo-related networks 
source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))
logger = logging.getLogger(__name__)


def construct_fake_data(env_stats, aid):
    basic_shape = (1, 1, env_stats.n_units)
    shapes = env_stats.obs_shape[aid]
    dtypes = env_stats.obs_dtype[aid]
    data = {k: jnp.zeros((*basic_shape, *v), dtypes[k]) 
        for k, v in shapes.items()}
    data = dict2AttrDict(data)
    action_dim = env_stats.action_dim[aid]
    data.action = jnp.zeros((*basic_shape, action_dim), jnp.float32)

    return data


def get_ith_model_prefix(i):
    return f'emodels/model{i}'


class Model(ModelBase):
    def add_attributes(self):
        self.elite_indices = None
        self.elite_idx = None
        self.n_selected_elites = collections.defaultdict(lambda: 0)

    def build_nets(self):
        aid = self.config.get('aid', 0)
        self.is_action_discrete = self.env_stats.is_action_discrete[aid]
        data = construct_fake_data(self.env_stats, aid)

        self.params.model, self.modules.model = self.build_net(
            data.obs[:, 0, :], data.action[:, 0], name='model')
        self.params.emodels, self.modules.emodels = self.build_net(
            data.obs, data.action, name='emodels')
        self.params.reward, self.modules.reward = self.build_net(
            data.obs, data.action, name='reward')
        self.params.discount, self.modules.discount = self.build_net(
            data.obs, name='discount')

    @property
    def theta(self):
        return self.params.subdict('emodels', 'reward', 'discount')

    def rank_elites(self, elite_indices):
        self.elite_indices = elite_indices

    def choose_elite(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.config.n_elites)
        self.elite_idx = self.elite_indices[idx]
        self.n_selected_elites[f'elite{self.elite_idx}'] += 1
        model = self.get_ith_model(self.elite_idx)
        assert set(model) == set(self.params.model), (set(model), set(self.params.model))
        self.params.model = model
        return self.elite_idx
    
    def evolve_model(self, score):
        indices = np.argsort(score)
        # replace the worst model with the best one
        model = self.get_ith_model(indices[-1], f'model/model{indices[0]}')
        # perturb the model with Gaussian noise
        for k, v in model.items():
            init = nn.initializers.normal(self.config.rn_std, dtype=jnp.float32)
            self.act_rng, rng = random.split(self.act_rng)
            model[k] = v + init(rng, v.shape)
        keys = set(self.params.emodels)
        self.params.emodels.update(model)
        assert keys == set(self.params.emodels), (keys, set(self.params.emodels))

    def get_ith_model(self, i, new_prefix='model/model'):
        model = {k.replace(get_ith_model_prefix(i), new_prefix): v 
            for k, v in self.params.emodels.items() 
            if k.startswith(get_ith_model_prefix(i))
        }
        return model

    def action(self, data, evaluation):
        self.act_rng, act_rng = jax.random.split(self.act_rng) 
        env_out, stats, state = self.jit_action(
            self.params, act_rng, data, evaluation)
        stats.update(self.n_selected_elites)
        return env_out, stats, state

    def raw_action(
        self, 
        params, 
        rng, 
        data, 
        evaluation=False, 
    ):
        rngs = random.split(rng, 3)
        action = self.process_action(data.action)

        stats = AttrDict()
        next_obs, stats = self.next_obs(
            params.model, rngs[0], data.obs, action, stats, evaluation)
        reward, stats = self.reward(
            params.reward, rngs[1], data.obs, action, stats)
        discount, stats = self.discount(
            params.discount, rngs[2], next_obs, stats)
        reset = 1 - discount
        if self.config.global_state_type == 'concat':
            global_state = jnp.expand_dims(next_obs, -3)
            global_state = jnp.reshape(global_state, (*global_state.shape[:-2], -1))
            global_state = jnp.tile(global_state, (self.env_stats.n_units, 1))
        elif self.config.global_state_type == 'obs':
            global_state = next_obs
        else:
            raise NotImplementedError
        obs = dict2AttrDict({'obs': next_obs, 'global_state': global_state})
        env_out = EnvOutput(obs, reward, discount, reset)

        return env_out, stats, data.state

    def next_obs(self, params, rng, obs, action, stats, evaluation):
        rngs = random.split(rng, 2)
        dist = self.modules.model(params, rngs[0], obs, action)
        if self.config.stoch_trans and not evaluation:
            next_obs = dist.sample(seed=rngs[1])
        else:
            next_obs = dist.mode()
        if isinstance(dist, jax_dist.MultivariateNormalDiag):
            # for continuous obs, we predict 𝛥(o)
            next_obs = obs + next_obs
        stats.update(dist.get_stats('model'))

        return next_obs, stats

    def reward(self, params, rng, obs, action, stats):
        dist = self.modules.reward(params, rng, obs, action)
        rewards = dist.mode()
        if isinstance(dist, jax_dist.Categorical):
            rewards = self.env_stats.reward_map[rewards]
        stats.update(dist.get_stats('reward'))

        return rewards, stats

    def discount(self, params, rng, obs, stats):
        dist = self.modules.discount(params, rng, obs)
        discount = dist.mode()
        stats.update(dist.get_stats('discount'))

        return discount, stats

    def process_action(self, action):
        if self.env_stats.is_action_discrete[0]:
            action = nn.one_hot(action, self.env_stats.action_dim[0])
        return action


def setup_config_from_envstats(config, env_stats):
    if 'aid' in config:
        aid = config['aid']
        config.emodels.out_size = env_stats.obs_shape[aid]['obs'][0]
    else:
        config.emodels.out_size = env_stats.obs_shape['obs'][0]
    config.model = config.emodels.copy()
    config.model.nn_id = 'model'
    config.model.pop('n_models')

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
        **kwargs
    )
