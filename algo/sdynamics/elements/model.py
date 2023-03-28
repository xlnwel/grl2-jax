import os
import numpy as np
import logging
import collections
import jax
from jax import nn, random
import jax.numpy as jnp
import enum

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


class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class Model(ModelBase):
    def add_attributes(self):
        self.elite_indices = np.arange(self.config.emodels.n_models)
        self.elite_idx = None
        self.n_elites = min(self.config.n_elites, self.config.emodels.n_models)
        self.n_selected_elites = collections.defaultdict(lambda: 0)

        if self.config.model_norm_obs:
            self.env_state_rms = RunningMeanStd([0, 1])

    def build_nets(self):
        aid = self.config.get('aid', 0)
        self.is_action_discrete = self.env_stats.is_action_discrete[aid]
        data = construct_fake_data(self.env_stats, aid)

        self.params.model, self.modules.model = self.build_net(
            data.env_state[:, 0, :], data.action[:, 0], name='model')
        self.params.emodels, self.modules.emodels = self.build_net(
            data.env_state, data.action, True, name='emodels')
        self.params.reward, self.modules.reward = self.build_net(
            data.env_state, data.action, data.env_state, name='reward')
        self.params.discount, self.modules.discount = self.build_net(
            data.env_state, name='discount')

    @property
    def theta(self):
        return self.params.subdict('emodels', 'reward', 'discount')

    def rank_elites(self, elite_indices):
        self.elite_indices = elite_indices

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
            data.env_state_loc, data.env_state_scale = self.env_state_rms.get_rms_stats(with_count=False)
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
        action = self.process_action(data.action)
        if self.config.model_norm_obs:
            data.env_state = normalize(data.env_state, data.env_state_loc, data.env_state_scale)
        
        if elite_indices is None:
            model = params.model
        else:
            idx = random.randint(rngs[0], (), 0, self.n_elites)
            model = self.get_ith_model(idx, eparams=params.emodels)
        stats = AttrDict()
        next_env_state, stats = self.next_env_state(
            model, rngs[0], data.env_state, action, stats, evaluation
        )
        reward, stats = self.reward(
            params.reward, rngs[1], data.env_state, action, next_env_state, stats
        )
        discount, stats = self.discount(
            params.discount, rngs[2], next_env_state, stats
        )
        reset = 1 - discount
        
        if self.config.model_norm_obs:
            next_env_state = denormalize(next_env_state, data.env_state_loc, data.env_state_scale)

        # TODO: here we output obs and global_state from env_state
        # here we output obs and global_state from env_state
        # if self.config.global_state_type == 'concat':
        #     global_state = jnp.expand_dims(next_obs, -3)
        #     global_state = jnp.reshape(global_state, (*global_state.shape[:-2], -1))
        #     global_state = jnp.tile(global_state, (self.env_stats.n_units, 1))
        # elif self.config.global_state_type == 'obs':
        #     global_state = next_obs
        # else:
        #     raise NotImplementedError
        next_obs = jnp.apply_along_axis(-1, next_env_state)
        print(next_env_state.shape)
        assert 0

        obs = dict2AttrDict({'env_state': next_env_state})
        # obs = dict2AttrDict({'obs': next_obs, 'global_state': global_state, 'env_state': env_state})
        env_out = EnvOutput(obs, reward, discount, reset)

        return env_out, stats, data.state

    def next_env_state(self, params, rng, env_state, action, stats, evaluation):
        rngs = random.split(rng, 2)

        dist = self.modules.model(params, rngs[0], env_state, action)
        if evaluation or self.config.deterministic_trans:
            next_env_state = dist.mode()
        else:
            next_env_state = dist.sample(seed=rngs[1])
        if isinstance(dist, jax_dist.MultivariateNormalDiag):
            next_env_state = env_state + next_env_state
        stats.update(dist.get_stats('model'))
        
        return next_env_state, stats

    def reward(self, params, rng, env_state, action, next_env_state, stats):
        dist = self.modules.reward(params, rng, env_state, action, next_env_state)
        rewards = dist.mode()
        if isinstance(dist, jax_dist.Categorical):
            rewards = self.env_stats.reward_map[rewards]
        stats.update(dist.get_stats('reward'))
        
        return rewards, stats

    def discount(self, params, rng, env_state, stats):
        dist = self.modules.discount(params, rng, env_state)
        discount = dist.mode()
        stats.update(dist.get_stats('discount'))

        return discount, stats

    def process_action(self, action):
        if self.env_stats.is_action_discrete[0]:
            action = nn.one_hot(action, self.env_stats.action_dim[0])
        return action 

    def get_env_state_rms_dir(self):
        path = '/'.join([self.config.root_dir, self.config.model_name])
        return path

    def save(self):
        super().save()
        self.save_env_state_rms()
    
    def restore(self):
        super().restore()
        self.restore_env_state_rms()

    def save_env_state_rms(self):
        filedir = self.get_env_state_rms_dir()
        save(self.popart, filedir=filedir, filename='popart')

    def restore_env_state_rms(self):
        filedir = self.get_env_state_rms_dir()
        self.popart = restore(
            filedir=filedir, filename='popart', 
            default=RunningMeanStd((0, 1)))

    ##### following is the codes for projecting env_state to global_state and obs
    def infer_global_state(self, state_vec, agent_id):
        # parse the state information
        state_dict = self.parse_state(state_vec)
        return self.env_state2global_state(state_dict, agent_id)

    def infer_obs(self, state_vec, agent_id):
        state_dict = self.parse_state(state_vec)
        return self.env_state2obs(state_dict, agent_id)

    def parse_state(self, state_vec):
        # parse the state information
        state_nf_al = 4 + self.env_stats.shield_bits_ally + self.env_stats.unit_type_bits
        state_nf_en = 4 + self.env_stats.shield_bits_enemy + self.env_stats.unit_type_bits
        last_action, state_timestep = None, None

        al_state = state_vec[:state_nf_al * self.env_stats.n_agents]
        ind = state_nf_al * self.env_stats.n_agents
        en_state = state_vec[ind:ind + state_nf_en * self.n_enemies]
        ind += state_nf_en * self.env_stats.n_enemies
        if self.env_stats.state_last_action:
            last_action = state_vec[ind:ind + self.env_stats.n_actions * self.env_stats.n_agents]
            ind += self.env_stats.n_actions * self.env_stats.n_agents
        if self.env_stats.state_timestep_number:
            state_timestep = state_vec[ind:ind + 1]
        al_feats = al_state.split(self.env_stats.n_agents)    # [n_agents, ally_feats]
        en_feats = en_state.split(self.env_stats.n_enemies)   # [n_enemies, enemy_feats]
        return {
            'al_feats': al_feats,
            'en_feats': en_feats,
            'last_action': last_action,
            'state_timestep': state_timestep,
        }

    def env_state2global_state(self, state_dict, agent_id):
        if self.env_stats.obs_instead_of_state or not self.env_stats.use_state_agent:
            raise NotImplementedError

        state_nf_en = state_nf_al = 5 + self.env_stats.unit_type_bits
        if self.env_stats.obs_all_health:
            state_nf_al += 1 + self.env_stats.shield_bits_ally
            state_nf_en += 1 + self.env_stats.shield_bits_enemy

        state_nf_own = 4 + self.env_stats.unit_type_bits
        if self.obs_own_health:
            state_nf_own += 1 + self.env_stats.shield_bits_ally

        if self.env_stats.obs_last_action:
            state_nf_al += self.env_stats.n_actions
            state_nf_own += self.env_stats.n_actions
        
        if self.env_stats.add_center_xy:
            state_nf_al += 2
            state_nf_en += 2
            state_nf_own += 2

        move_feats_len = self.env_stats.n_actions_move
        if self.env_stats.obs_pathing_grid:
            move_feats_len += self.env_stats.n_obs_pathing
        if self.env_stats.obs_terrain_height:
            move_feats_len += self.env_stats.n_obs_height

        move_feats = jnp.zeros(move_feats_len, dtype=jnp.float32)
        enemy_feats = jnp.zeros((self.env_stats.n_enemies, state_nf_en), dtype=jnp.float32)
        ally_feats = jnp.zeros((self.env_stats.n_agents - 1, state_nf_al), dtype=jnp.float32)
        own_feats = jnp.zeros(state_nf_own, dtype=jnp.float32)
        agent_id_feats = jnp.zeros(self.env_stats.n_agents, dtype=jnp.float32) 
        
        center_x = self.env_stats.map_x / 2
        center_y = self.env_stats.map_y / 2
        
        agent_state_feat = state_dict['al_feats'][agent_id]
        health, weapon, rel_x, rel_y  = agent_state_feat[:4]
        pos_x, pos_y = self.recover_pos(rel_x, rel_y)
        if self.env_stats.shield_bits_ally > 0:
            own_shield = agent_state_feat[4]
        if self.env_stats.unit_type_bits > 0:
            own_unit_type = agent_state_feat[-self.env_stats.unit_type_bits:]

        if health > 0:
            # compute movement feat
            if self.can_move(pos_x, pos_y, Direction.NORTH):
                move_feats[0] = 1
            if self.can_move(pos_x, pos_y, Direction.SOUTH):
                move_feats[1] = 1
            if self.can_move(pos_x, pos_y, Direction.EAST):
                move_feats[2] = 1
            if self.can_move(pos_x, pos_y, Direction.WEST):
                move_feats[3] = 1
            
            if self.env_stats.obs_pathing_grid or self.env_stats.obs_terrain_height:
                raise NotImplementedError

            # compute enemy features
            for e_id in range(self.n_enemies):
                enemy_state_feat = state_dict['en_feats'][e_id]
                en_health, en_rel_x, en_rel_y = enemy_state_feat[:3]
                if self.env_stats.shield_bits_enemy > 0:
                    en_shield = enemy_state_feat[3]
                if self.env_stats.unit_type_bits > 0:
                    en_unit_type = enemy_state_feat[-self.env_stats.unit_type_bits:]
                en_pos_x, en_pos_y = self.recover_pos(en_rel_x, en_rel_y)
                dist = self.distance(pos_x, pos_y, en_pos_x, en_pos_y)

                if en_health > 0:   # only alive required
                    # visible and alive
                    enemy_feats[e_id, 0] = float(dist <= self.env_stats.shoot_range)
                    enemy_feats[e_id, 1] = dist / self.sight_range
                    enemy_feats[e_id, 2] = (en_pos_x - pos_x) / self.sight_range
                    enemy_feats[e_id, 3] = (en_pos_y - pos_y) / self.sight_range
                    enemy_feats[e_id, 4] = float(dist < self.env_stats.sight_range)

                    ind = 5
                    if self.obs_all_health:
                        enemy_feats[e_id, ind] = en_health
                        ind += 1
                        if self.env_stats.shield_bits_enemy > 0:
                            enemy_feats[e_id, ind] = en_shield
                            ind += 1
                    if self.env_stats.unit_type_bits > 0:
                        enemy_feats[e_id, ind:ind+self.env_stats.unit_type_bits] = en_unit_type
                        ind += self.env_stats.unit_type_bits
                    if self.env_stats.add_center_xy:
                        enemy_feats[e_id, ind] = (en_pos_x - center_x) / self.env_stats.max_distance_x
                        enemy_feats[e_id, ind+1] = (en_pos_y - center_y) / self.env_stats.max_distance_y
            
            # compute ally features
            al_ids = [
                al_id for al_id in range(self.n_agents) if al_id != agent_id
            ]
            for i, al_id in enumerate(al_ids):
                ally_state_feat = state_dict['al_feats'][al_id]
                al_health, al_weapon, al_rel_x, al_rel_y = ally_state_feat[:4]
                al_pos_x, al_pos_y = self.recover_pos(al_rel_x, al_rel_y)
                if self.env_stats.shield_bits_ally > 0:
                    al_shield = ally_state_feat[4]
                if self.env_stats.unit_type_bits > 0:
                    al_unit_type = ally_state_feat[-self.env_stats.unit_type_bits:]
                dist = self.distance(pos_x, pos_y, al_pos_x, al_pos_y)
                if al_health > 0:
                    ally_feats[i, 0] = float(dist < self.env_stats.sight_range)
                    ally_feats[i, 1] = dist / self.sight_range
                    ally_feats[i, 2] = (al_pos_x - pos_x) / self.sight_range
                    ally_feats[i, 3] = (al_pos_y - pos_y) / self.sight_range
                    ally_feats[i, 4] = al_weapon

                    ind = 5
                    if self.env_stats.obs_all_health:
                        ally_feats[i, ind] = al_health
                        ind += 1
                        if self.env_stats.shield_bits_ally > 0:
                            ally_feats[i, ind] = al_shield
                            ind += 1
                    
                    if self.env_stats.add_center_xy:
                        ally_feats[i, ind] = (al_pos_x - center_x) / self.env_stats.max_distance_x
                        ally_feats[i, ind+1] = (al_pos_y - center_y) / self.env_stats.max_distance_y
                        ind += 2

                    if self.env_stats.unit_type_bits > 0:
                        ally_feats[al_id, ind:ind+self.env_stats.unit_type_bits] = al_unit_type
                        ind += self.env_stats.unit_type_bits
                    
                    if self.env_stats.obs_last_action:
                        ally_feats[al_id, ind:] = state_dict['last_action'][al_id*self.env_stats.n_actions:(al_id+1)*self.env_stats.n_actions]
                
            # own features
            own_feats[0] = 1    # visible
            own_feats[1] = 0    # distance
            own_feats[2] = 0    # relative X
            own_feats[3] = 0    # relative Y
            ind = 4
            if self.obs_own_health:
                own_feats[ind] = health
                ind += 1
                if self.env_stats.shield_bits_ally > 0:
                    own_feats[ind] = own_shield
                    ind += 1
            if self.env_stats.add_center_xy:
                own_feats[ind] = (pos_x - center_x) / self.env_stats.max_distance_x
                own_feats[ind+1] = (pos_y - center_y) / self.env_stats.max_distance_y
                ind += 2
            if self.env_stats.unit_type_bits > 0:
                own_feats[ind:ind+self.env_stats.unit_type_bits] = own_unit_type
                ind += self.unit_type_bits
            if self.state_last_action:
                own_feats[ind:] = state_dict['last_action'][agent_id*self.env_stats.n_actions:(agent_id+1)*self.env_stats.n_actions]

        state = jnp.concatenate((ally_feats.flatten(),
                                enemy_feats.flatten(),
                                move_feats.flatten(),
                                own_feats.flatten()))

        # agent id features
        if self.state_agent_id:
            agent_id_feats[agent_id] = 1.
            state = jnp.concatenate((state, agent_id_feats.flatten()))
            
        if self.state_timestep_number:
            state = jnp.concatenate((state, self._episode_steps / self.episode_limit))
        
        return state

    def env_state2obs(self, state_dict, agent_id):
        # define some properties about the obersvation
        obs_nf_en = obs_nf_al = 4 + self.env_stats.unit_type_bits
        if self.obs_all_health:
            obs_nf_al += 1 + self.env_stats.shield_bits_ally
            obs_nf_en += 1 + self.env_stats.shield_bits_enemy
        
        obs_nf_own = 4 + self.env_stats.unit_type_bits
        if self.obs_own_health:
            obs_nf_own += 1 + self.env_stats.shield_bits_ally

        if self.obs_last_action:
            obs_nf_al += self.env_stats.n_actions
            obs_nf_own += self.env_stats.n_actions
        
        move_feats_len = self.env_stats.n_actions_move
        if self.env_stats.obs_pathing_grid:
            move_feats_len += self.env_stats.n_obs_pathing
        if self.env_stats.obs_terrain_height:
            move_feats_len += self.env_stats.n_obs_height

        move_feats = jnp.zeros(move_feats_len, dtype=jnp.float32)
        enemy_feats = jnp.zeros((self.env_stats.n_enemies, obs_nf_en), dtype=jnp.float32)
        ally_feats = jnp.zeros((self.env_stats.n_agents - 1, obs_nf_al), dtype=jnp.float32)
        own_feats = jnp.zeros(obs_nf_own, dtype=jnp.float32)
        agent_id_feats = jnp.zeros(self.env_stats.n_agents, dtype=jnp.float32)

        agent_state_feat = state_dict['al_feats'][agent_id]
        health, weapon, rel_x, rel_y  = agent_state_feat[:4]
        pos_x, pos_y = self.recover_pos(rel_x, rel_y)
        if self.env_stats.shield_bits_ally > 0:
            own_shield = agent_state_feat[4]
        if self.env_stats.unit_type_bits > 0:
            own_unit_type = agent_state_feat[-self.env_stats.unit_type_bits:]

        if health > 0:
            # compute movement feat
            if self.can_move(pos_x, pos_y, Direction.NORTH):
                move_feats[0] = 1
            if self.can_move(pos_x, pos_y, Direction.SOUTH):
                move_feats[1] = 1
            if self.can_move(pos_x, pos_y, Direction.EAST):
                move_feats[2] = 1
            if self.can_move(pos_x, pos_y, Direction.WEST):
                move_feats[3] = 1
            
            if self.env_stats.obs_pathing_grid or self.env_stats.obs_terrain_height:
                raise NotImplementedError
            
            # compute enemy features
            for e_id in range(self.n_enemies):
                enemy_state_feat = state_dict['en_feats'][e_id]
                en_health, en_rel_x, en_rel_y = enemy_state_feat[:3]
                if self.env_stats.shield_bits_enemy > 0:
                    en_shield = enemy_state_feat[3]
                if self.env_stats.unit_type_bits > 0:
                    en_unit_type = enemy_state_feat[-self.env_stats.unit_type_bits:]
                en_pos_x, en_pos_y = self.recover_pos(en_rel_x, en_rel_y)
                dist = self.distance(pos_x, pos_y, en_pos_x, en_pos_y)

                if (
                    dist < self.sight_range and en_health > 0
                ):
                    # visible and alive
                    enemy_feats[e_id, 0] = float(dist <= self.shoot_range)
                    enemy_feats[e_id, 1] = dist / self.sight_range
                    enemy_feats[e_id, 2] = (en_pos_x - pos_x) / self.sight_range
                    enemy_feats[e_id, 3] = (en_pos_y - pos_y) / self.sight_range

                    ind = 4
                    if self.obs_all_health:
                        enemy_feats[e_id, ind] = en_health
                        ind += 1
                        if self.env_stats.shield_bits_enemy > 0:
                            enemy_feats[e_id, ind] = en_shield
                            ind += 1
                    if self.env_stats.unit_type_bits > 0:
                        enemy_feats[e_id, ind:] = en_unit_type 

            # compute ally features
            al_ids = [
                al_id for al_id in range(self.n_agents) if al_id != agent_id
            ]
            for i, al_id in enumerate(al_ids):
                ally_state_feat = state_dict['al_feats'][al_id]
                al_health, al_weapon, al_rel_x, al_rel_y = ally_state_feat[:4]
                al_pos_x, al_pos_y = self.recover_pos(al_rel_x, al_rel_y)
                if self.env_stats.shield_bits_ally > 0:
                    al_shield = ally_state_feat[4]
                if self.env_stats.unit_type_bits > 0:
                    al_unit_type = ally_state_feat[-self.env_stats.unit_type_bits:]
                dist = self.distance(pos_x, pos_y, al_pos_x, al_pos_y)
                if (
                    dist < self.sight_range and al_health > 0
                ):
                    ally_feats[i, 0] = 1
                    ally_feats[i, 1] = dist / self.sight_range
                    ally_feats[i, 2] = (al_pos_x - pos_x) / self.sight_range
                    ally_feats[i, 3] = (al_pos_y - pos_y) / self.sight_range

                    ind = 4
                    if self.env_stats.obs_all_health:
                        ally_feats[i, ind] = al_health
                        ind += 1
                        if self.env_stats.shield_bits_ally > 0:
                            ally_feats[i, ind] = al_shield
                            ind += 1
                    if self.env_stats.unit_type_bits > 0:
                        ally_feats[al_id, ind:ind+self.env_stats.unit_type_bits] = al_unit_type
                        ind += self.env_stats.unit_type_bits
                    
                    if self.env_stats.obs_last_action:
                        ally_feats[al_id, ind:] = state_dict['last_action'][al_id*self.env_stats.n_actions:(al_id+1)*self.env_stats.n_actions]

            # own features
            own_feats[0] = 1    # visible
            own_feats[1] = 0    # distance
            own_feats[2] = 0    # relative X
            own_feats[3] = 0    # Y
            ind = 4
            if self.obs_own_health:
                own_feats[ind] = health
                ind += 1
                if self.env_stats.shield_bits_ally > 0:
                    own_feats[ind] = own_shield
                    ind += 1
            if self.env_stats.unit_type_bits > 0:
                own_feats[ind:ind+self.env_stats.unit_type_bits] = own_unit_type
                ind += self.unit_type_bits
            if self.obs_last_action:
                own_feats[ind:] = state_dict['last_action'][agent_id*self.env_stats.n_actions:(agent_id+1)*self.env_stats.n_actions]

        agent_obs = jnp.concatenate(
            (
                move_feats.flatten(),
                enemy_feats.flatten(),
                ally_feats.flatten(),
                own_feats.flatten(),
            )
        )
        
        if self.obs_agent_id:
            agent_id_feats[agent_id] = 1
            agent_obs = jnp.concatenate((agent_obs, agent_id_feats))
        
        if self.obs_timestep_number:
            assert state_dict['state_timestep'] is not None
            agent_obs = jnp.concatenate((agent_obs, state_dict['state_timestep']))

        return agent_obs

    def can_move(self, pos_x, pos_y, direction):
        m = self.env_stats.move_amount / 2
        
        if direction == Direction.NORTH:
            x, y = int(pos_x), int(pos_y + m)
        elif direction == Direction.SOUTH:
            x, y = int(pos_x), int(pos_y - m)
        elif direction == Direction.EAST:
            x, y = int(pos_x + m), int(pos_y)
        else:
            x, y = int(pos_x - m), int(pos_y)

        check_bounds = lambda x, y: (0 <= x < self.env_stats.map_x and 0 <= y < self.env_stats.map_y)
        if check_bounds(x, y) and self.env_stats.pathing_grid[x, y]:
            return True

        return False
    
    def recover_pos(self, rel_x, rel_y):
        center_x, center_y = self.env_stats.map_x / 2, self.env_stats.map_y / 2
        return rel_x * self.env_stats.max_distance_x + center_x, rel_y * self.env_stats.max_distance_y + center_y

    @staticmethod 
    def distance(x1, y1, x2, y2):
        return jnp.hypot(x2 - x1, y2 - y1)

    def sanity_check(self, env_kwargs):
        assert env_kwargs['obs_all_health']
        assert env_kwargs['obs_own_health']
        assert env_kwargs['obs_last_action']
        assert not env_kwargs['obs_pathing_grid']
        assert not env_kwargs['obs_terrain_height']
        assert not env_kwargs['obs_timestep_number']
        assert not env_kwargs['obs_instead_of_state']
        assert env_kwargs['state_last_action']
        assert not env_kwargs['state_timestep_number']


def setup_config_from_envstats(config, env_stats):
    if 'aid' in config:
        aid = config['aid']
        config.emodels.out_size = env_stats.obs_shape[aid]['env_state'][0]
    else:
        config.emodels.out_size = env_stats.obs_shape['env_state'][0]
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