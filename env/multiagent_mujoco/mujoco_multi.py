from functools import partial
import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
import numpy as np

from .multiagentenv import MultiAgentEnv
from .obsk import get_joints_at_kdist, get_parts_and_edges, build_obs

# using code from https://github.com/ikostrikov/pytorch-ddpg-naf
class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        action = (action + 1) / 2
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def action(self, action_):
        return self._action(action_)

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action


class MujocoMulti(MultiAgentEnv):

    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        self.scenario = kwargs["env_args"]["scenario"]  # e.g. Ant-v2
        self.agent_conf = kwargs["env_args"]["agent_conf"]  # e.g. '2x3'

        self.agent_partitions, self.mujoco_edges, self.mujoco_globals = get_parts_and_edges(self.scenario,
                                                                                             self.agent_conf)

        self.n_agents = len(self.agent_partitions)
        self.n_actions = max([len(l) for l in self.agent_partitions])
        self.obs_add_global_pos = kwargs["env_args"].get("obs_add_global_pos", False)

        self.agent_obsk = kwargs["env_args"].get("agent_obsk", None) # if None, fully observable else k>=0 implies observe nearest k agents or joints
        self.agent_obsk_agents = kwargs["env_args"].get("agent_obsk_agents", False)  # observe full k nearest agents (True) or just single joints (False)

        if self.agent_obsk is not None:
            self.k_categories_label = kwargs["env_args"].get("k_categories")
            if self.k_categories_label is None:
                if self.scenario in ["Ant-v2", "manyagent_ant"]:
                    self.k_categories_label = "qpos,qvel,cfrc_ext|qpos"
                elif self.scenario in ["Humanoid-v2", "HumanoidStandup-v2"]:
                    self.k_categories_label = "qpos,qvel,cfrc_ext,cvel,cinert,qfrc_actuator|qpos"
                elif self.scenario in ["Reacher-v2"]:
                    self.k_categories_label = "qpos,qvel,fingertip_dist|qpos"
                elif self.scenario in ["coupled_half_cheetah"]:
                    self.k_categories_label = "qpos,qvel,ten_J,ten_length,ten_velocity|"
                else:
                    self.k_categories_label = "qpos,qvel|qpos"

            k_split = self.k_categories_label.split("|")
            self.k_categories = [k_split[k if k < len(k_split) else -1].split(",") for k in range(self.agent_obsk+1)]

            self.global_categories_label = kwargs["env_args"].get("global_categories")
            self.global_categories = self.global_categories_label.split(",") if self.global_categories_label is not None else []


        if self.agent_obsk is not None:
            self.k_dicts = [get_joints_at_kdist(agent_id,
                                                self.agent_partitions,
                                                self.mujoco_edges,
                                                k=self.agent_obsk,
                                                kagents=False,) for agent_id in range(self.n_agents)]

        # load scenario from script
        self.episode_limit = self.args.episode_limit
        self.max_episode_steps = self.args.episode_limit

        self.env_version = kwargs["env_args"].get("env_version", 2)
        if self.env_version == 2:
            try:
                self.wrapped_env = NormalizedActions(gym.make(self.scenario))
            except gym.error.Error:  # env not in gym
                if self.scenario in ["manyagent_ant"]:
                    from .manyagent_ant import ManyAgentAntEnv as this_env
                elif self.scenario in ["manyagent_swimmer"]:
                    from .manyagent_swimmer import ManyAgentSwimmerEnv as this_env
                elif self.scenario in ["coupled_half_cheetah"]:
                    from .coupled_half_cheetah import CoupledHalfCheetah as this_env
                else:
                    raise NotImplementedError('Custom env not implemented!')
                self.wrapped_env = NormalizedActions(TimeLimit(this_env(**kwargs["env_args"]), max_episode_steps=self.episode_limit))
        else:
            assert False,  "not implemented!"
        self.timelimit_env = self.wrapped_env.env
        self.timelimit_env._max_episode_steps = self.episode_limit
        self.env = self.timelimit_env.env
        self.timelimit_env.reset()
        self.obs_size = self.get_obs_size()

        # COMPATIBILITY
        self.n = self.n_agents
        self.observation_space = [Box(low=np.array([-10]*self.n_agents), high=np.array([10]*self.n_agents)) for _ in range(self.n_agents)]

        self.obs_shape = [{
            'obs': (self.obs_size, ), 
            'global_state': (self.obs_size, )
        } for _ in range(self.n)]
        self.obs_dtype = [{
            'obs': np.float32, 
            'global_state': np.float32
        } for _ in range(self.n)]
        self.uid2aid = list(range(self.n_agents))

        acdims = [len(ap) for ap in self.agent_partitions]
        self.action_space = tuple([Box(self.env.action_space.low[sum(acdims[:a]):sum(acdims[:a+1])],
                                       self.env.action_space.high[sum(acdims[:a]):sum(acdims[:a+1])]) for a in range(self.n_agents)])
        
        self.reward_range = None
        self.metadata = None
        self._score = np.zeros(self.n)
        self._dense_score = np.zeros(self.n)
        self._epslen = 0

    def random_action(self):
        action = [a.sample() for a in self.action_space]
        return action

    def step(self, actions):

        # we need to map actions back into MuJoCo action space
        actions = np.reshape(actions, (self.n, -1))
        env_actions = np.zeros((sum([self.action_space[i].low.shape[0] for i in range(self.n_agents)]),)) + np.nan
        for a, partition in enumerate(self.agent_partitions):
            for i, body_part in enumerate(partition):
                if env_actions[body_part.act_ids] == env_actions[body_part.act_ids]:
                    raise Exception("FATAL: At least one env action is doubly defined!")
                env_actions[body_part.act_ids] = actions[a][i]

        if np.isnan(env_actions).any():
            raise Exception("FATAL: At least one env action is undefined!")

        obs_n, reward_n, done_n, info_n = self.wrapped_env.step(env_actions)
        reward = [np.ones(1) * reward_n for _ in range(self.n)]
        self.steps += 1
        self._score += reward_n
        self._dense_score += reward_n
        self._epslen = self.steps

        info = {
            'score': self._score, 
            'dense_score': self._dense_score, 
            'epslen': self._epslen, 
            'game_over': self._epslen == self.max_episode_steps
        }

        if done_n and self._epslen == self.max_episode_steps:
            done = [np.zeros(1) for _ in range(self.n)]
        else:
            done = [np.ones(1) * done_n for _ in range(self.n)]

        return self.get_obs(), reward, done, info

    def get_obs(self):
        """ Returns all agent observat3ions in a list """
        obs_n = []
        state = self.env._get_obs()
        for a in range(self.n_agents):
            agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
            agent_id_feats[a] = 1.0
            obs_i = np.concatenate([state, agent_id_feats])
            obs_i = (obs_i - np.mean(obs_i)) / np.std(obs_i)
            obs_i = np.expand_dims(obs_i, 0)
            obs_n.append({'obs': obs_i, 'global_state': obs_i})
        return obs_n

    def get_obs_agent(self, agent_id):
        if self.agent_obsk is None:
            return self.env._get_obs()
        else:
            return build_obs(self.env,
                                  self.k_dicts[agent_id],
                                  self.k_categories,
                                  self.mujoco_globals,
                                  self.global_categories,
                                  vec_len=getattr(self, "obs_size", None))

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self.env._get_obs()) + self.n_agents
        # if self.agent_obsk is None:
        #     return self.get_obs_agent(0).size
        # else:
        #     return max([len(self.get_obs_agent(agent_id)) for agent_id in range(self.n_agents)])


    def get_state(self, team=None):
        # TODO: May want global states for different teams (so cannot see what the other team is communicating e.g.)
        return self.env._get_obs()

    def get_state_size(self):
        """ Returns the shape of the state"""
        return len(self.get_state())

    def get_avail_actions(self): # all actions are always available
        return np.ones(shape=(self.n_agents, self.n_actions,))

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(shape=(self.n_actions,))

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions # CAREFUL! - for continuous dims, this is action space dim rather
        # return self.env.action_space.shape[0]

    def get_stats(self):
        return {}

    # TODO: Temp hack
    def get_agg_stats(self, stats):
        return {}

    def reset(self, **kwargs):
        """ Returns initial observations and states"""
        self.steps = 0
        self._score = np.zeros(self.n)
        self._dense_score = np.zeros(self.n)
        self._epslen = self.steps
        self.timelimit_env.reset()
        return self.get_obs()

    def render(self, **kwargs):
        self.env.render()
    
    def get_screen(self):
        img = self.env.render(mode='rgb_array')
        return img

    def close(self):
        raise NotImplementedError

    def seed(self, args):
        pass

    def get_env_info(self):

        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    "action_spaces": self.action_space,
                    "actions_dtype": np.float32,
                    "normalise_actions": False
                    }
        return env_info
