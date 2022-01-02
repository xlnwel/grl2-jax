import numpy as np
import gym

from env.overcooked_env.mdp.actions import Action
from env.overcooked_env.mdp.overcooked_mdp import OvercookedGridworld
from env.overcooked_env.mdp.overcooked_env import OvercookedEnv
from env.overcooked_env.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from utility.utils import AttrDict2dict


REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 1,
    "DISH_PICKUP_REWARD": 1,
    "SOUP_PICKUP_REWARD": 1,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0
}


class Overcooked:
    def __init__(self, config):
        config = AttrDict2dict(config)
        self.name = config['env_name'].split('-', 1)[-1]
        self._mdp = OvercookedGridworld.from_layout_name(
            layout_name=self.name, **config.get('layout_params', {}), rew_shaping_params=REW_SHAPING_PARAMS)
        self._env = OvercookedEnv.from_mdp(self._mdp, horizon=config['max_episode_steps'], info_level=0)

        self._featurize = config.get('featurize', False)
        if self._featurize:
            mlp = MediumLevelActionManager.from_pickle_or_compute(self._mdp, NO_COUNTERS_PARAMS)
            self.featurize_fn = lambda x: np.stack(self._mdp.featurize_state(x, mlp))

        if self._featurize:
            self.obs_shape = dict(
                obs=self._setup_observation_shape(),
                global_state=self._setup_global_state_shape(),
            )
            self.obs_dtype = dict(
                obs=np.float32,
                global_state=np.float32
            )
        else:
            self.obs_shape = dict(
                obs=self._setup_observation_shape(),
            )
            self.obs_dtype = dict(
                obs=np.uint8,
            )
        
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.action_dim = self.action_space.n
        self.max_episode_steps = config['max_episode_steps']
        self.pid2aid = config.get('pid2aid', [0, 1])
        self.n_players = len(self.pid2aid)
        self.n_agents = len(set([aid for aid in self.pid2aid]))
        self.dense_reward = config.get('dense_reward', False)

    def get_screen(self, **kwargs):
        """
        Standard way to view the state of an esnvironment programatically
        is just to print the Env object
        """
        return self._env.__repr__()
    
    def random_action(self):
        return np.random.randint(0, self.action_dim, self.n_players)

    def _setup_observation_shape(self):
        dummy_state = self._mdp.get_standard_start_state()
        if self._featurize:
            obs_shape = self._env.featurize_state_mdp(dummy_state)[0].shape
            return obs_shape
        else:    
            self._env.reset()
            obs_shape = self._env.lossless_state_encoding_mdp(dummy_state)[0].shape
            return obs_shape

    def _setup_global_state_shape(self):
        return self._setup_observation_shape()
    
    def reset(self):
        self._env.reset()
        obs = self.get_obs(self._env.state)
        self._score = np.zeros(self.n_players, dtype=np.float32)
        self._dense_score = np.zeros(self.n_players, dtype=np.float32)
        self._epslen = 0

        return obs

    def step(self, action):
        assert action.shape == (2,), action
        action = Action.ALL_ACTIONS[action[0]], Action.ALL_ACTIONS[action[1]]
        state, reward, done, info = self._env.step(action)
        rewards = reward * np.ones(self.n_players, np.float32)
        self._score += rewards
        self._epslen += 1
        if self.dense_reward and reward == 0:
            dense_reward = max(info['shaped_r_by_agent'])
            rewards = dense_reward * np.ones(self.n_players, np.float32)
        self._dense_score += rewards
        obs = self.get_obs(state)
        dones = np.array([done, done], dtype=np.bool)
        info.update(dict(
            score=self._score,
            epslen=self._epslen,
            dense_score=self._dense_score,
            game_over=done
        ))

        return obs, rewards, dones, info

    def get_obs(self, state):
        if self._featurize:
            obs = self._env.featurize_state_mdp(state)
            return dict(
                obs=obs,
                global_state=obs,
            )
        else:
            return dict(
                obs=[obs.astype(np.uint8) 
                    for obs in self._env.lossless_state_encoding_mdp(state)]
            )

    def close(self):
        pass


if __name__ == '__main__':
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('env', type=str)
        parser.add_argument('--interactive', '-i', action='store_true')
        args = parser.parse_args()
        return args
    args = parse_args()
    config = dict(
        env_name=args.env,
        max_episode_steps=400,
        dense_reward=True,
        featurize=False,
        # layout_params={
        #     'onion_time': 1,
        # }
    )
    def action2char(action):
        dic = {
            'w': (0, -1),
            's': (0, 1),
            'a': (-1, 0),
            'd': (1, 0),
            'q': (0, 0),
            'e': 'interact',
        }
        a1, a2 = dic[action[0]], dic[action[1]]
        return Action.ACTION_TO_CHAR[a1], Action.ACTION_TO_CHAR[a2]
    def action2array(action):
        dic = {
            'w': 0,
            's': 1,
            'a': 3,
            'd': 2,
            'q': 4,
            'e': 5,
        }
        return np.array([dic[action[0]], dic[action[1]]])

    env = Overcooked(config)
    obs = env.reset()
    d = False
    while not np.all(d):
        print(env.get_screen())
        if args.interactive:
            a = input('action: ').split(' ')
        else:
            a = env.random_action()
        print(action2char(a))
        o, r, d, i = env.step(action2array(a))
        print(o['obs'][0][0, 0])
        print(o['obs'][0][3, 0])
        print(o['obs'][0][4, 0])
        print("Curr reward: (sparse)", i['sparse_r_by_agent'], "\t(dense)", i['shaped_r_by_agent'])
        print('Reward', r)
