import numpy as np
import gym

from env.overcooked_env.mdp.actions import Action
from env.overcooked_env.mdp.overcooked_mdp import OvercookedGridworld
from env.overcooked_env.mdp.overcooked_env import OvercookedEnv
from env.overcooked_env.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from env.utils import compute_aid2pids
from utility.utils import AttrDict2dict


REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 1,
    "DISH_PICKUP_REWARD": 0,
    "SOUP_PICKUP_REWARD": 0,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0
}


def compute_pots_ingradients(terrain):
    pots_pos = []
    n_pots = 0
    n_onions = 0
    n_tomatoes = 0
    for y, row in enumerate(terrain):
        for x, i in enumerate(row):
            if i == 'P':
                n_pots += 1
                pots_pos.append((x, y))
            elif i == 'O':
                n_onions += 1
            elif i == 'T':
                n_tomatoes += 1
    assert len(pots_pos) == n_pots
    return pots_pos, n_pots, n_onions, n_tomatoes


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

        self.pots_pos, self.n_pots, self.n_onions, self.n_tomatoes = compute_pots_ingradients(self._mdp.terrain_mtx)
        self.goal_size = self.n_pots * 2

        self.pid2aid = config.get('pid2aid', [0, 1])
        self.aid2pids = compute_aid2pids(self.pid2aid)
        self.n_players = len(self.pid2aid)
        self.n_agents = len(self.aid2pids)

        if self._featurize:
            self.obs_shape = [dict(
                obs=self._setup_observation_shape(aid),
                global_state=self._setup_global_state_shape(aid),
            ) for aid in range(self.n_agents)]
            self.obs_dtype = [dict(
                obs=np.float32,
                global_state=np.float32,
            ) for _ in self.aid2pids]
        else:
            self.obs_shape = [dict(
                obs=self._setup_observation_shape(aid),
            ) for aid in range(self.n_agents)]
            self.obs_dtype = [dict(
                obs=np.float32,
            ) for _ in self.aid2pids]
        self._add_goal = config.get('add_goal', False)
        if self._add_goal:
            for shape, dtype in zip(self.obs_shape, self.obs_dtype):
                shape['goal'] = (self.goal_size,)
                dtype['goal'] = np.float32

        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.action_shape = self.action_space.shape
        self.action_dim = self.action_space.n
        self.action_dtype = np.int32
        self.max_episode_steps = config['max_episode_steps']
        self.dense_reward = config.get('dense_reward', False)

    def get_screen(self, **kwargs):
        """
        Standard way to view the state of an esnvironment programatically
        is just to print the Env object
        """
        return self._env.__repr__()
    
    def random_action(self):
        return np.random.randint(0, self.action_dim, self.n_players)

    def _setup_observation_shape(self, aid):
        dummy_state = self._mdp.get_standard_start_state()
        if self._featurize:
            obs_shape = self._env.featurize_state_mdp(dummy_state)[0].shape
            return obs_shape
        else:    
            self._env.reset()
            obs_shape = self._env.lossless_state_encoding_mdp(dummy_state)[0].shape
            return obs_shape

    def _setup_global_state_shape(self, aid):
        return self._setup_observation_shape(aid)
    
    def reset(self):
        self._env.reset()
        obs = self.get_obs(self._env.state)
        self._score = np.zeros(self.n_players, dtype=np.float32)
        self._dense_score = np.zeros(self.n_players, dtype=np.float32)
        self._epslen = 0

        return obs

    def step(self, action):
        assert action.shape == (2,), action
        real_action = Action.ALL_ACTIONS[action[0]], Action.ALL_ACTIONS[action[1]]
        state, reward, done, info = self._env.step(real_action)
        rewards = reward * np.ones(self.n_players, np.float32)
        self._score += rewards
        self._epslen += 1
        if self.dense_reward:
            dense_reward = max(info['shaped_r_by_agent'])
            rewards += dense_reward * np.ones(self.n_players, np.float32)
        # else:
        #     print(reward, info['sparse_r_by_agent'])
        self._dense_score += rewards
        obs = self.get_obs(state, action)
        dones = np.array([done, done], dtype=np.bool)
        info.update(dict(
            score=self._score,
            epslen=self._epslen,
            dense_score=self._dense_score,
            game_over=done
        ))

        return obs, rewards, dones, info

    def get_obs(self, state, action=None):
        if self._featurize:
            obs = self._env.featurize_state_mdp(state)
            obs = dict(
                obs=obs,
                global_state=obs,
            )
        else:
            obs = dict(
                obs=[obs.astype(np.float32) 
                    for obs in self._env.lossless_state_encoding_mdp(state)],
            )
        if self._add_goal:
            goal = self.get_pots_status()
            obs['goal'] = [goal for _ in range(self.n_agents)]
        return obs

    def get_pots_status(self):
        goal = np.ones(self.goal_size, np.float32)
        for i, pos in enumerate(self.pots_pos):
            if pos in self._env.state.objects:
                soup = self._env.state.objects[pos]
                for x in soup.ingredients:
                    if x == 'tomato':
                        goal[2*i] -= 1
                    elif x == 'onion':
                        goal[2*i+1] -= 1
        return goal

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
        add_goal=True
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
        print(o['goal'])
        print("Curr reward: (sparse)", i['sparse_r_by_agent'], "\t(dense)", i['shaped_r_by_agent'])
        print('Reward', r)
