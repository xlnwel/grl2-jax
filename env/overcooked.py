import numpy as np
import gym

from env.overcooked_env.mdp.actions import Action
from env.overcooked_env.mdp.overcooked_mdp import OvercookedGridworld
from env.overcooked_env.mdp.overcooked_env import OvercookedEnv
from env.overcooked_env.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from utility.utils import AttrDict2dict


BASE_REW_SHAPING_PARAMS = {
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
            layout_name=self.name, **config.get('layout_params', {}), rew_shaping_params=BASE_REW_SHAPING_PARAMS)
        self._env = OvercookedEnv.from_mdp(self._mdp, horizon=config['max_episode_steps'])

        mlp = MediumLevelActionManager.from_pickle_or_compute(self._mdp, NO_COUNTERS_PARAMS)
        self.featurize_fn = lambda x: np.stack(self._mdp.featurize_state(x, mlp))

        self.obs_shape = dict(
            obs=self._setup_observation_shape(),
            global_state=self._setup_global_state_shape(),
        )
        self.obs_dtype = dict(
            obs=np.float32,
            global_state=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.action_dim = self.action_space.n
        self.max_episode_steps = config['max_episode_steps']
        self.n_players = 2
        self.pid2aid = [0, 1]
        self.dense_reward = config.get('dense_reward', False)

    def __repr__(self):
        """
        Standard way to view the state of an esnvironment programatically
        is just to print the Env object
        """
        return self._env.__repr__()
    
    def random_action(self):
        return np.random.randint(0, self.action_dim, self.n_players)

    def _setup_observation_shape(self):
        dummy_state = self._mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        return obs_shape

    def _setup_global_state_shape(self):
        dummy_state = self._mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        return obs_shape
    
    def reset(self):
        self._env.reset()
        obs = self.featurize_fn(self._env.state)
        obs = dict(
            obs=obs,
            global_state=obs,
        )

        self._score = np.zeros(self.n_players, dtype=np.float32)
        self._dense_score = np.zeros(self.n_players, dtype=np.float32)
        self._epslen = 0

        return obs
    
    def step(self, action):
        assert action.shape == (2,), action
        action = Action.ALL_ACTIONS[action[0]], Action.ALL_ACTIONS[action[1]]
        state, reward, done, info = self._env.step(action)
        self._score += reward
        self._epslen += 1
        reward = np.array([reward for _ in range(self.n_players)], np.float32)
        if self.dense_reward:
            reward += np.array(info['shaped_r_by_agent'], dtype=np.float32)
        self._dense_score += reward
        obs = self.featurize_fn(state)
        obs = dict(
            obs=obs,
            global_state=obs,
        )
        dones = np.array([done, done], dtype=np.bool)
        info.update(dict(
            score=self._score,
            epslen=self._epslen,
            dense_score=self._dense_score,
            game_over=done
        ))

        return obs, reward, dones, info

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
        layout_params={
            'onion_time': 1,
        }
    )
    env = Overcooked(config)
    obs = env.reset()
    d = False
    while not np.all(d):
        if args.interactive:
            a = np.array(eval(input('action: ')))
        else:
            a = env.random_action()
        print(Action.ACTION_TO_CHAR[a[0]], Action.ACTION_TO_CHAR[a[1]])
        o, r, d, i = env.step(a)
        print(env)
        print("Curr reward: (sparse)", i['sparse_r_by_agent'], "\t(dense)", i['shaped_r_by_agent'])
