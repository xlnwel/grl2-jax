import os
import numpy as np
import logging
import collections
import jax
from jax import nn, random
import jax.numpy as jnp

from core.ckpt.pickle import save, restore
from core.elements.model import Model as ModelBase
from core.typing import dict2AttrDict, AttrDict
from tools.file import source_file
from jax_tools import jax_dist
from env.typing import EnvOutput
from tools.rms import *
from .utils import *

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
        self.elite_indices = np.arange(self.config.emodels.n_models)
        self.elite_idx = None
        self.n_elites = min(self.config.n_elites, self.config.emodels.n_models)
        self.n_selected_elites = collections.defaultdict(lambda: 0)

        self.obs_rms = RunningMeanStd([0, 1])

    def build_nets(self):
        aid = self.config.get('aid', 0)
        self.is_action_discrete = self.env_stats.is_action_discrete[aid]
        data = construct_fake_data(self.env_stats, aid)

        self.params.model, self.modules.model = self.build_net(
            data.obs[:, 0, :], data.action[:, 0], name='model')
        self.params.emodels, self.modules.emodels = self.build_net(
            data.obs, data.action, True, name='emodels')
        self.params.reward, self.modules.reward = self.build_net(
            data.obs, data.action, name='reward')
        self.params.discount, self.modules.discount = self.build_net(
            data.obs, data.action, name='discount')

    @property
    def theta(self):
        return self.params.subdict('emodels', 'reward', 'discount')

    def rank_elites(self, metrics):
        self.elite_indices = np.argsort(metrics)
        self.choose_elite()

    def choose_elite(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.n_elites)
        self.elite_idx = self.elite_indices[idx]
        self.n_selected_elites[f'elite{self.elite_idx}'] += 1
        model = self.get_ith_model(self.elite_idx)
        assert set(model) == set(self.params.model), (set(self.params.model) - set(model))
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

    def get_ith_model(self, i, new_prefix='model/model', eparams=None):
        if eparams is None:
            eparams = self.params.emodels
        model = {k.replace(get_ith_model_prefix(i), new_prefix): v 
            for k, v in eparams.items() if k.startswith(get_ith_model_prefix(i))
        }
        return model

    def action(self, data, evaluation):
        self.act_rng, act_rng = jax.random.split(self.act_rng)
        if self.config.model_norm_obs:
            data.obs_loc, data.obs_scale = self.obs_rms.get_rms_stats(with_count=False)
        data.dim_mask = self.get_const_dim_mask()
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
        elite_indices=None, 
    ):
        rngs = random.split(rng, 3)
        if data.dim_mask is None:
            dim_mask = jnp.ones_like(data.obs)
        else:
            dim_mask = jnp.zeros_like(data.obs) + data.dim_mask
        action = self.process_action(data.action)
        if self.config.model_norm_obs:
            obs = normalize(
                data.obs, 
                data.obs_loc, 
                data.obs_scale, 
                dim_mask=dim_mask, 
                np=jnp
            )
        else:
            obs = data.obs

        if elite_indices is None:
            model = self.modules.model
            model_params = params.model
        else:
            model = self.modules.emodels
            model_params = params.emodels
        stats = AttrDict()
        next_obs, stats = self.next_obs(
            model, model_params, rngs[0], obs, action, 
            dim_mask, elite_indices, stats, evaluation
        )
        
        reward_obs = self.get_reward_obs(dim_mask, data.obs, obs)
        reward, stats = self.reward(
            params.reward, rngs[1], reward_obs, action, stats)
        
        discount, stats = self.discount(
            params.discount, rngs[2], obs, action, stats)

        if self.config.model_norm_obs:
            next_obs = denormalize(
                next_obs, 
                data.obs_loc, 
                data.obs_scale, 
                dim_mask=dim_mask, 
                np=jnp
            )
        next_obs = jnp.where(dim_mask, next_obs, data.obs)

        if self.config.global_state_type == 'concat':
            global_state = jnp.expand_dims(next_obs, -3)
            global_state = jnp.reshape(global_state, (*global_state.shape[:-2], -1))
            global_state = jnp.tile(global_state, (self.env_stats.n_units, 1))
        elif self.config.global_state_type == 'obs':
            global_state = next_obs
        else:
            raise NotImplementedError

        obs = dict2AttrDict({'obs': next_obs, 'global_state': global_state})
        
        # Deal with the case when the environment has already been reset
        prev_discount = 1 - data.reset
        prev_discount_exp = jnp.expand_dims(prev_discount, -1)
        obs = jax.tree_util.tree_map(lambda x: x * prev_discount_exp, obs)
        obs.sample_mask = prev_discount
        reward = reward * prev_discount
        discount = discount * prev_discount
        reset = 1 - discount
        
        env_out = EnvOutput(obs, reward, discount, reset)

        return env_out, stats, data.state

    def next_obs(self, model, params, rng, obs, action, 
                 dim_mask, elite_indices, stats, evaluation):
        rngs = random.split(rng, 3)

        dist = model(params, rngs[0], obs, action)
        if evaluation or self.config.deterministic_trans:
            next_obs = dist.mode()
        else:
            next_obs = dist.sample(seed=rngs[1])
        if elite_indices is not None:
            i = random.choice(rngs[2], elite_indices)
            next_obs = next_obs.take(i, axis=0)
        if isinstance(dist, jax_dist.MultivariateNormalDiag):
            # for continuous obs, we predict 𝛥(o)
            next_obs = obs + next_obs
        next_obs = jnp.where(dim_mask, next_obs, obs)

        return next_obs, stats

    def reward(self, params, rng, obs, action, stats=AttrDict()):
        dist = self.modules.reward(params, rng, obs, action)
        rewards = dist.mode()
        if isinstance(dist, jax_dist.Categorical):
            rewards = self.env_stats.reward_map[rewards]

        return rewards, stats

    def discount(self, params, rng, obs, action, stats):
        dist = self.modules.discount(params, rng, obs, action)
        discount = dist.mode()

        return discount, stats

    def process_action(self, action):
        if self.env_stats.is_action_discrete[0]:
            action = nn.one_hot(action, self.env_stats.action_dim[0])
        return action

    def get_reward_obs(self, dim_mask, obs, norm_obs):
        # For agent-specific reward, we keep constant dimensions in that some dimensions represent the agent's identity
        reward_obs = jnp.where(dim_mask, norm_obs, obs)
        return reward_obs

    def get_discount_obs(self, dim_mask, pred_obs):
        discount_next_obs = jnp.where(dim_mask, pred_obs, 0)
        return discount_next_obs
        
    def get_obs_rms_dir(self):
        path = '/'.join([self.config.root_dir, self.config.model_name])
        return path

    def get_obs_rms(self, with_count=False, return_std=True):
        return self.obs_rms.get_rms_stats(with_count=with_count, return_std=return_std) 

    def get_const_dim_mask(self):
        dim_mask = self.obs_rms.const_dim_mask()
        return np.any(dim_mask, 0)

    def update_obs_rms(self, rms):
        if rms is not None:
            assert self.config.model_norm_obs, self.config.model_norm_obs
            self.obs_rms.update_from_moments(*rms)

    def save(self):
        super().save()
        self.save_obs_rms()
    
    def restore(self):
        super().restore()
        self.restore_obs_rms()

    def save_obs_rms(self):
        filedir = self.get_obs_rms_dir()
        save(self.obs_rms, filedir=filedir, filename='obs_rms')

    def restore_obs_rms(self):
        filedir = self.get_obs_rms_dir()
        self.obs_rms = restore(
            filedir=filedir, filename='obs_rms', 
            default=RunningMeanStd((0, 1))
        )


def setup_config_from_envstats(config, env_stats):
    if 'aid' in config:
        aid = config['aid']
        config.emodels.out_size = env_stats.obs_shape[aid]['obs'][0]
    else:
        config.emodels.out_size = env_stats.obs_shape['obs'][0]
    config.model = config.emodels.copy()
    config.model.nn_id = 'model'
    config.model.pop('n_models')
    if config.model_loss_type == 'mse':
        # for MSE loss, we only consider deterministic transitions since the variance is unconstrained.
        config.deterministic_trans = True

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
