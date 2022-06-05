import numpy as np
import gym
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from env.utils import *


class OpenSpiel:
    def __init__(
        self, 
        env_name, 
        uid2aid=None, 
        **kwargs, 
    ):
        self.env = env = rl_environment.Environment(env_name)
        self.uid2aid = uid2aid
        self.aid2uids = compute_aid2uids(self.uid2aid)
        self.n_units = len(self.uid2aid)
        self.n_agents = len(self.aid2uids)

        state_size = env.observation_spec()["info_state"][0]
        num_actions = env.action_spec()["num_actions"]
        self.action_space = [
            gym.spaces.Discrete(num_actions) for _ in self.aid2uids
        ]
        self.action_shape = [a.shape for a in self.action_space]
        self.action_dim = [a.n for a in self.action_space]
        self.action_dtype = [np.int32 for _ in self.action_space]
        self.is_action_discrete = [isinstance(a, gym.spaces.Discrete) for a in self.action_space]

        self.obs_shape = [dict(
            obs=(state_size, ), 
            global_state=(state_size, ), 
            prev_reward=(), 
            prev_action=(ad,), 
            action_mask=(ad, )
        ) for ad in self.action_dim]
        self.obs_dtype = [dict(
            obs=np.float32, 
            global_state=np.float32, 
            prev_reward=np.float32, 
            prev_action=np.float32, 
            action_mask=bool
        ) for _ in range(self.n_agents)]

        self.use_action_mask = True

    @property
    def current_player(self):
        return self._current_player

    def random_action(self):
        return [np.random.choice(a) 
            for a in self._time_step.observations['legal_actions']
                if a != []
        ]

    def seed(self, seed=None):
        print('seed', seed)
        self.env.seed(seed)

    def reset(self):
        self._time_step = self.env.reset()
        self._current_player = self._time_step.observations['current_player']

        return self._get_obs(self._time_step)

    def step(self, action):
        assert action[0] in self._time_step.observations['legal_actions'][self._current_player], \
            (action, self._time_step.observations['legal_actions'][self._current_player])
        self._time_step = self.env.step(action)
        if self._time_step.observations['current_player'] >= 0:
            self._current_player = self._time_step.observations['current_player']

        rewards = np.array(self._time_step.rewards)
        discounts = np.array(self._time_step.discounts, dtype=np.float32)

        if self._time_step.last():
            assert np.all(discounts == 0), discounts

        info = {
            'game_over': self._time_step.last()
        }

        obs = self._get_obs(self._time_step)

        return obs, rewards, discounts, info

    def _get_obs(self, time_step):
        uid = max(time_step.observations['current_player'], 0)
        info_state = np.array(time_step.observations['info_state'][uid], dtype=np.float32)
        action_mask = np.zeros(self.action_dim[uid], bool)
        legal_action = time_step.observations['legal_actions'][uid]
        action_mask[legal_action] = 1
        obs = dict(
            obs=info_state, 
            global_state=info_state, 
            action_mask=action_mask,
            uid=uid
        )

        return obs
    
    def seed(self, seed=None):
        self.env.seed(seed)

    def close(self):
        return


if __name__ == "__main__":
    config = dict(
        env_name='spiel-leduc_poker',
        squeeze_keys=['uid'], 
        uid2aid=[0, 1],
        n_envs=2
    )
    # from env.func import create_env
    # env = create_env(config)
    # done = np.zeros((config['n_envs'], len(config['uid2aid'])), bool)
    # reward = np.zeros((config['n_envs'], len(config['uid2aid'])))
    # for i in range(10):
    #     print('iteration', i)
    #     a = env.random_action()
    #     agent_outs = env.step(a)
    #     print('action\t', np.squeeze(a))
    #     for pid, out in enumerate(agent_outs):
    #         if len(out.obs) == 0:
    #             continue
    #         uid = out.obs['current_player']
    #         print(np.stack([np.arange(len(uid)), uid]))
    #         uid = tuple(zip(np.arange(len(uid)), uid))
    #         print(out.reward)
    #         print(out.reward[uid])
    config['env_name'] = 'leduc_poker'
    env = OpenSpiel(**config)
    obs = env.reset()
    print(obs['uid'])
    for i in range(10):
        a = env.random_action()
        obs, reward, discount, info = env.step(a)
        if np.all(discount == 0):
            obs = env.reset()
            print(obs['uid'])