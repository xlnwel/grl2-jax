import numpy as np
import gym
import gfootball.env as football_env

from env.utils import *


class GRF:
    def __init__(
        self,
        # built-in configs for grf
        env_name,
        representation='simple115v2',
        rewards='scoring,checkpoints',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False,
        write_video=False,
        dump_frequency=1000,
        logdir='data/grf',
        extra_players=None,
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,
        # custom grf configs
        shared_ckpt_reward=False, 
        # required configs for grl
        max_episode_steps=3000,
        use_action_mask=False,
        uid2aid=None,
        seed=None, 
        **kwargs,
    ):
        self.name = env_name
        assert number_of_left_players_agent_controls in (1, 11), \
            number_of_left_players_agent_controls
        assert number_of_right_players_agent_controls in (0, 1, 11), \
            number_of_right_players_agent_controls

        if uid2aid is None:
            if number_of_left_players_agent_controls > 0:
                uid2aid = tuple(np.zeros(number_of_left_players_agent_controls, dtype=np.int32)) \
                    + tuple(np.ones(number_of_right_players_agent_controls, dtype=np.int32))
            else:
                uid2aid = tuple(np.zeros(number_of_right_players_agent_controls, dtype=np.int32))
        self.uid2aid = uid2aid
        self.aid2uids = compute_aid2uids(self.uid2aid)
        self.n_units = len(self.uid2aid)
        self.n_agents = len(self.aid2uids)

        self.number_of_left_players_agent_controls = number_of_left_players_agent_controls
        self.number_of_right_players_agent_controls = number_of_right_players_agent_controls

        assert number_of_left_players_agent_controls + number_of_right_players_agent_controls \
            == self.n_units, \
                (self.uid2aid, number_of_left_players_agent_controls, number_of_right_players_agent_controls)

        self.shared_ckpt_reward = shared_ckpt_reward
        if self.shared_ckpt_reward and rewards == 'scoring,checkpoints':
            rewards = 'scoring'
        other_config_options = {} if seed is None else {'seed': seed}
        # print('other config options', other_config_options)
        self.env = football_env.create_environment(
            self.name, 
            representation=representation,
            rewards=rewards,
            write_goal_dumps=write_goal_dumps,
            write_full_episode_dumps=write_full_episode_dumps,
            render=render,
            write_video=write_video,
            dump_frequency=dump_frequency,
            logdir=logdir,
            extra_players=extra_players,
            number_of_left_players_agent_controls=number_of_left_players_agent_controls,
            number_of_right_players_agent_controls=number_of_right_players_agent_controls,
            other_config_options=other_config_options
        )

        self.max_episode_steps = max_episode_steps

        self.use_action_mask = use_action_mask  # if action mask is used
        self.use_life_mask = False              # if life mask is used

        self.action_space = [
            self.env.action_space[0] 
            if isinstance(self.env.action_space, gym.spaces.MultiDiscrete) 
            else self.env.action_space 
            for _ in range(self.n_agents)]
        self.action_shape = [a.shape for a in self.action_space]
        self.action_dim = [a.n for a in self.action_space]
        self.action_dtype = [np.int32 for _ in self.action_space]
        self.is_action_discrete = [isinstance(a, gym.spaces.Discrete) for a in self.action_space]

        self.obs_shape = self._get_observation_shape()
        self.obs_dtype = self._get_observation_dtype()

        # The following stats should be updated in self.step and be reset in self.reset
        # The episodic score we use to evaluate agent's performance. It excludes shaped rewards
        self._score = np.zeros(self.n_units, dtype=np.float32)
        # The accumulated episodic rewards we give to the agent. It includes shaped rewards
        self._dense_score = np.zeros(self.n_units, dtype=np.float32)
        # The length of the episode
        self._epslen = 0
        self._left_score = np.zeros(self.n_units, dtype=np.float32)
        self._right_score = np.zeros(self.n_units, dtype=np.float32)

        self._prev_action = [-1 for _ in self.action_dim]
        self._consecutive_action = np.zeros(self.n_units, bool)

        self._checkpoint_reward = .1
        self._num_checkpoints = 10
        self._collected_checkpoints = [0, 0]

    def _get_observation_shape(self):
        obs_shape = self.env.observation_space.shape \
            if self.n_units == 1 else self.env.observation_space.shape[1:]
        shape = [dict(
            obs=obs_shape, 
            global_state=obs_shape, 
            prev_reward=(), 
            prev_action=(self.action_dim[i],), 
        ) for i in range(self.n_agents)]

        if self.use_action_mask:
            for aid in range(self.n_agents):
                shape[aid]['action_mask'] = (self.action_space[aid].n,)

        return shape

    def _get_observation_dtype(self):
        obs_dtype = self.env.observation_space.dtype
        dtype = [dict(
            obs=obs_dtype,
            global_state=obs_dtype,
            prev_reward=np.float32, 
            prev_action=np.float32, 
        ) for _ in range(self.n_agents)]

        if self.use_action_mask:
            for aid in range(self.n_agents):
                dtype[aid]['action_mask'] = bool

        return dtype

    def random_action(self):
        action = [
            np.random.randint(0, self.action_dim[0], len(uids)) 
            for uids in self.aid2uids
        ]

        return action

    def reset(self):
        obs = self.env.reset()

        self._score = np.zeros(self.n_units, dtype=np.float32)
        self._dense_score = np.zeros(self.n_units, dtype=np.float32)
        self._epslen = 0
        self._left_score = np.zeros(self.n_units, dtype=np.float32)
        self._right_score = np.zeros(self.n_units, dtype=np.float32)
        self._ckpt_score = np.zeros(self.n_units, dtype=np.float32)

        self._prev_action = [-1 for _ in self.action_dim]
        self._consecutive_action = np.zeros(self.n_units, bool)
        self._collected_checkpoints = [0, 0]

        return self._get_obs(obs)

    def step(self, action):
        action_oh = [np.zeros((len(uids), a), np.float32) 
            for uids, a in zip(self.aid2uids, self.action_dim)]
        for uids, a, oh in zip(self.aid2uids, action, action_oh):
            oh[np.arange(len(uids)), a] = 1
        action = np.concatenate(action)
        obs, reward, done, info = self.env.step(action)

        reward = self._get_reward(reward, info)
        if self.number_of_right_players_agent_controls == 0:
            self._ckpt_score += reward - info['score_reward']
        else:
            self._ckpt_score[:self.number_of_left_players_agent_controls] += \
                reward[:self.number_of_left_players_agent_controls] - info['score_reward']
            self._ckpt_score[self.number_of_left_players_agent_controls:] += \
                reward[self.number_of_left_players_agent_controls:] + info['score_reward']

        rewards = np.reshape(reward, -1)

        self._epslen += 1
        self._dense_score += rewards
        self._left_score += 1 if info['score_reward'] == 1 else 0
        self._right_score += 1 if info['score_reward'] == -1 else 0
        if self.name.startswith('11_vs_11') and self._epslen == self.max_episode_steps:
            done = True
            self._score = np.where(
                self._left_score < self._right_score, -1, 
                self._left_score > self._right_score)
            self._score[self.number_of_left_players_agent_controls:] = \
                - self._score[self.number_of_left_players_agent_controls:]
        dones = np.tile(done, self.n_units)

        self._consecutive_action = np.array(
            [pa == a for pa, a in zip(self._prev_action, action)], bool)
        self._prev_action = action
        diff_score = self._left_score - self._right_score
        if self.number_of_right_players_agent_controls != 0:
            diff_score[-self.number_of_right_players_agent_controls:] *= -1
        info = {
            'score': self._score,
            'dense_score': self._dense_score,
            'left_score': self._left_score,
            'right_score': self._right_score,
            'diff_score': diff_score,
            'consecutive_action': self._consecutive_action, 
            'checkpoint_score': self._ckpt_score, 
            'epslen': self._epslen,
            'game_over': done
        }

        agent_rewards = [np.reshape(rewards[uids], -1) for uids in self.aid2uids]
        agent_dones = [np.reshape(dones[uids], -1) for uids in self.aid2uids]
        agent_obs = self._get_obs(obs, action_oh, agent_rewards)

        return agent_obs, agent_rewards, agent_dones, info

    def close(self):
        return self.env.close()

    def _get_obs(self, obs, action=None, reward=None):
        if action is None:
            action = [np.zeros((len(uids), a), np.float32) 
                for uids, a in zip(self.aid2uids, self.action_dim)]
            reward = [np.zeros(len(uids), np.float32) for uids in self.aid2uids]

        if self.n_units == 1:
            obs = np.expand_dims(obs, 0)

        agent_obs = [dict(
            obs=obs[uids], 
            global_state=obs[uids], 
            prev_reward=reward[aid], 
            prev_action=action[aid], 
        ) for aid, uids in enumerate(self.aid2uids)]

        return agent_obs

    def _get_reward(self, reward, info):
        def add_ckpt_reward(reward, side):
            assert side in [0, -1], side
            if (info['score_reward'] == 1 and side == 0) or \
                    (info['score_reward'] == -1 and side == -1):
                reward += self._checkpoint_reward * (
                    self._num_checkpoints - self._collected_checkpoints[side])
                self._collected_checkpoints[side] = self._num_checkpoints
            else:
                o = self.env.unwrapped.observation()[side]
                if 'ball_owned_team' not in o or o['ball_owned_team'] != side:
                    return reward
                d = ((o['ball'][0] - 1) ** 2 + o['ball'][1] ** 2) ** 0.5
                while self._collected_checkpoints[side] < self._num_checkpoints:
                    threshold = (.99 - .8 / (self._num_checkpoints - 1)
                        * self._collected_checkpoints[side])
                    if d > threshold:
                        break
                    reward += self._checkpoint_reward
                    self._collected_checkpoints[side] += 1
            return reward

        if self.shared_ckpt_reward:
            if self.number_of_right_players_agent_controls == 0:
                reward = add_ckpt_reward(reward, 0)
            else:
                reward[:self.number_of_left_players_agent_controls] = add_ckpt_reward(
                    reward[:self.number_of_left_players_agent_controls], 0
                )
                reward[self.number_of_left_players_agent_controls:] = add_ckpt_reward(
                    reward[self.number_of_left_players_agent_controls:], -1
                )

        return reward

    def seed(self, seed):
        return seed

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', '-l', type=int, default=11)
    parser.add_argument('--right', '-r', type=int, default=0)
    parser.add_argument('--step', '-s', type=int, default=10)
    parser.add_argument('--unit', '-u', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = {
        'env_name': '11_vs_11_easy_stochastic',
        'rewards': 'scoring,checkpoints', 
        'number_of_left_players_agent_controls': args.left,
        'number_of_right_players_agent_controls': args.right,
        'shared_ckpt_reward': True, 
        'uid2aid': None,
        'seed': 1
    }

    from utility.display import print_dict_info, print_dict
    env = GRF(**config)
    env2 = GRF(**config)
    import random
    random.seed(0)
    np.random.seed(0)
    n = env.obs_shape[0]['obs'][0]
    o = env.reset()
    random.seed(0)
    np.random.seed(0)
    o2 = env2.reset()
    np.testing.assert_allclose(o[0]['obs'], o2[0]['obs'])
    for i in range(args.step):
        print(i)
        a = env.random_action()
        random.seed(0)
        np.random.seed(0)
        o, r, d, info = env.step(a)
        random.seed(0)
        np.random.seed(0)
        o2, r2, d2, info2 = env2.step(a)
        print(a)
        np.testing.assert_allclose(o[0]['obs'], o2[0]['obs'])
