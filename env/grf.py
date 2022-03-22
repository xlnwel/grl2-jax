import numpy as np
import gym
import gfootball.env as football_env

from env.utils import *


def _check_in_field(x, y):
    return (-1 <= x <= 1) and (-.42 <= y <= .42)

def _check_all_in_field(pos):
    return [_check_in_field(x, y) for x, y in pos]

def _get_roles(obs_list, uids):
    n = len(obs_list)
    roles = np.zeros((n, 10), dtype=np.float32)
    role_idx = obs_list[0]['left_team_roles'][uids]
    roles[np.arange(n), role_idx] = 1

    assert roles.shape == (n, 10), roles.shape

    return roles

def _get_ball_info(obs_list, n_left, n_right=0):
    def get_ball_info(obs):
        ball_owned_player = np.zeros(23, dtype=np.float32)
        ball_owned_player[obs['ball_owned_player']] = 1
        ball_info = np.concatenate([
            obs['ball'], 
            obs['ball_direction'], 
            obs['ball_rotation'], 
            ball_owned_player
        ])

        return ball_info

    ball_info = [get_ball_info(obs_list[0])] * n_left
    if n_right:
        right_ball_info = get_ball_info(obs_list[n_left])
        ball_info = ball_info + [right_ball_info] * n_right
    ball_info = np.stack(ball_info)

    assert ball_info.shape == (len(obs_list), 32), ball_info.shape

    return ball_info

def _get_player_info(obs_list, uids):
    pass

def _compute_relative_pos(obs_list, left_uids, right_uids):
    def compute_relative_pos(obs, uids):
        all_pos = np.concatenate([obs['left_team'], obs['right_team']], 0)
        unit_pos = all_pos[uids]
        rel_pos = np.expand_dims(unit_pos, 1) - np.expand_dims(all_pos, 0)
        return unit_pos, rel_pos
    
    unit_pos, rel_pos = compute_relative_pos(obs_list[0], left_uids)
    if len(right_uids) > 0:
        right_unit_pos, right_rel_pos = compute_relative_pos(obs_list[-1], right_uids)
        unit_pos = np.concatenate([unit_pos, right_unit_pos], 0)
        rel_pos = np.concatenate([rel_pos, right_rel_pos], 0)

    for o, up, rp in zip(obs_list, unit_pos, rel_pos):
        idx = o['active']
        pos = o['left_team'][idx]
        other_pos = np.concatenate([o['left_team'], o['right_team']], 0)
        assert other_pos.shape == (22, 2), other_pos.shape
        np.testing.assert_allclose(pos, up)
        np.testing.assert_allclose(pos-other_pos, rp)

    assert unit_pos.shape == (len(obs_list), 2), unit_pos.shape
    assert rel_pos.shape == (len(obs_list), 22, 2), rel_pos.shape

    return unit_pos, rel_pos

def _compute_units_info(obs, k, n_left, n_right):
    if k < n_left:
        t1 = 'left_team'
        t2 = 'right_team'
    else:
        t1 = 'right_team'
        t2 = 'left_team'
        k -= n_left
    pos = []
    directions = []
    tire_factor = []
    yellow_card = []
    active = []

    p = obs[t1][k]
    suffixes = ['', '_direction', '_tired_factor', '_yellow_card', '_active']
    for i, (pi, d, tf, yc, a) in enumerate(
        zip(*[np.concatenate([obs[f'{t1}{s}'], obs[f'{t2}{s}']]) for s in suffixes])):
        if i != k:
            pos.append(compute_relative_position(p, pi))
            directions.append(d)
            tire_factor.append(tf)
            yellow_card.append(yc)
            active.append(a)

    units = np.concatenate([
        pos, 
        directions, 
        np.expand_dims(tire_factor, -1), 
        np.expand_dims(yellow_card, -1), 
        np.expand_dims(active, -1)], 
        -1, dtype=np.float32)
    assert units.shape == (21, 7), (units.shape, n_left, n_right)

    return units

def _check_obs_consistency(obs, obs_list):
    # pass
    for team in ['right_team']:
        for name in ['']:
            # name = f'{team}{name}'
            data = obs_list[0][team]
            for i, o in enumerate(obs_list):
                actual = obs[i][44:66]
                # print(actual.reshape(-1, 2))
                target = o[team].reshape(-1)
                np.testing.assert_allclose(actual, target, err_msg=f'{name}_{i}')

def _divide_data(data, n_left, n_right):
    left_data = np.array(data[:n_left], dtype=np.float32)
    right_data = np.array(data[-n_right:], dtype=np.float32)

    return left_data, right_data

def _compute_all_obs(obs_list, n_left, n_right):
    assert len(obs_list) == n_left + n_right, (len(obs_list), n_left, n_right)

    units_info = []
    state_info = []
    for i, obs in enumerate(obs_list):
        game_mode = np.zeros(7, np.float32)
        game_mode[obs['game_mode']] = 1
        # ball_owned_team = np.zeros(22, np.float32)
        # ball_owned_team[obs['ball_owned_team']] = 1
        # ball_owned_player = np.zeros(22, np.float32)
        # ball_owned_player[obs['ball_owned_player']] = 1
        # print(i, 'ball_owned_player', obs['ball_owned_team'], obs['ball_owned_player'])
        pos = obs['left_team' if i < 11 else 'right_team'][i%11]
        ball_pos = obs['ball']
        ball_rp = compute_relative_position(pos, ball_pos[:2])
        ball_cs = compute_angle_cos_sin(pos, ball_pos[:2])

        units_info.append(_compute_units_info(obs, i, n_left, n_right))
        state_info.append(np.concatenate([
            pos,
            obs['sticky_actions'], 
            [obs['designated'], 
            obs['steps_left']],
            obs['score'], 
            ball_rp, 
            ball_cs, 
            obs['ball_direction'], 
            obs['ball_rotation'], 
            [ball_pos[-1],
            obs['ball_owned_team'], 
            ],
            game_mode, 
        ]))
    if n_left > 0 and n_right > 0:
        left_units_info = np.array(units_info[:n_left], dtype=np.float32)
        right_units_info = np.array(units_info[-n_right:], dtype=np.float32)
        left_state_info = np.array(state_info[:n_left], dtype=np.float32)
        right_state_info = np.array(state_info[-n_right:], dtype=np.float32)

        obs = [dict(
            units=units,
            own_state=state,
        ) for units, state in [
            [left_units_info, left_state_info], 
            [right_units_info, right_state_info]
            ]
        ]
    else:
        obs = [dict(
            units=np.array(units_info, dtype=np.float32), 
            own_state=np.array(state_info, dtype=np.float32), 
        )]

    return obs


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
        unit_based_obs=False, 
        add_role_to_obs=False, 
        infield_penalty=0, 
        # required configs for grl
        max_episode_steps=3000,
        use_action_mask=True,
        uid2aid=None,
        **kwargs,
    ):
        self.name = env_name
        assert number_of_left_players_agent_controls == 1 \
            or number_of_left_players_agent_controls == 11, \
                number_of_left_players_agent_controls
        assert number_of_right_players_agent_controls == 0 \
            or number_of_right_players_agent_controls == 1 \
            or number_of_right_players_agent_controls == 11, \
                number_of_right_players_agent_controls

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
        )

        self.infield_penalty = infield_penalty

        self.max_episode_steps = max_episode_steps

        self.use_action_mask = use_action_mask  # if action mask is used
        self.use_life_mask = False              # if life mask is used
        self.unit_based_obs = unit_based_obs
        self.add_role_to_obs = add_role_to_obs

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
        self._infield_score = np.zeros(self.n_units, dtype=np.float32)

        self._prev_action = [-1 for _ in self.action_dim]
        self._consecutive_action = np.zeros(self.n_units, bool)

    def _get_observation_shape(self):
        obs_shape = self.env.observation_space.shape \
            if self.n_units == 1 else self.env.observation_space.shape[1:]
        if self.unit_based_obs:
            shape = [dict(
                units=(21, 17), 
                state=(27,), 
                prev_reward=(), 
                prev_action=(self.action_dim[i],), 
            ) for i in range(self.n_agents)]
        else:
            shape = [dict(
                obs=obs_shape, 
                global_state=obs_shape, 
                prev_reward=(), 
                prev_action=(self.action_dim[i],), 
            ) for i in range(self.n_agents)]
        if self.add_role_to_obs:
            for s in shape:
                s['role'] = (10,)

        if self.use_action_mask:
            for aid in range(self.n_agents):
                shape[aid]['action_mask'] = (self.action_space[aid].n,)

        return shape

    def _get_observation_dtype(self):
        obs_dtype = self.env.observation_space.dtype
        if self.unit_based_obs:
            dtype = [dict(
                units=np.float32,
                state=np.float32,
                prev_reward=np.float32, 
                prev_action=np.float32, 
            ) for _ in range(self.n_agents)]
        else:
            dtype = [dict(
                obs=obs_dtype,
                global_state=obs_dtype,
                prev_reward=np.float32, 
                prev_action=np.float32, 
            ) for _ in range(self.n_agents)]
        if self.add_role_to_obs:
            for t in dtype:
                t['role'] = np.float32

        if self.use_action_mask:
            for aid in range(self.n_agents):
                dtype[aid]['action_mask'] = bool

        return dtype

    def random_action(self):
        action = [[] for _ in self.aid2uids]
        for aid in self.uid2aid:
            action[aid].append(self.action_space[0].sample())
        action = [np.array(a, dtype=np.int32) for a in action]

        return action

    def reset(self):
        obs = self.env.reset()

        self._score = np.zeros(self.n_units, dtype=np.float32)
        self._dense_score = np.zeros(self.n_units, dtype=np.float32)
        self._epslen = 0
        self._left_score = np.zeros(self.n_units, dtype=np.float32)
        self._right_score = np.zeros(self.n_units, dtype=np.float32)
        self._infield_score = np.zeros(self.n_units, dtype=np.float32)

        self._prev_action = [-1 for _ in self.action_dim]
        self._consecutive_action = np.zeros(self.n_units, bool)

        return self._get_obs(obs)

    def step(self, action):
        action_oh = [np.zeros((len(uids), a), np.float32) for uids, a in zip(self.aid2uids, self.action_dim)]
        for uids, a, oh in zip(self.aid2uids, action, action_oh):
            oh[np.arange(len(uids)), a] = 1
        action = np.concatenate(action)
        obs, reward, done, info = self.env.step(action)

        rewards = np.reshape(reward, -1)
        if self.infield_penalty != 0:
            assert self.infield_penalty < 0, self.infield_penalty
            idx = np.argmax(obs[:, 97: 108], -1)
            idx[self.number_of_left_players_agent_controls:] += 11
            pos = np.concatenate([obs[0, :22], obs[0, 44:66]]).reshape(-1, 2)
            pos = pos[idx]
            infield = 1-np.array(_check_all_in_field(pos), np.float32)
            infield_penalty = np.where(infield, self.infield_penalty, 0)
            self._infield_score += infield_penalty
            rewards += infield_penalty

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

        self._consecutive_action = np.array([pa == a for pa, a in zip(self._prev_action, action)], bool)
        self._prev_action = action
        info = {
            'score': self._score,
            'dense_score': self._dense_score,
            'left_score': self._left_score,
            'right_score': self._right_score,
            'diff_score': self._left_score - self._right_score,
            'infield_score': self._infield_score, 
            'consecutive_action': self._consecutive_action, 
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

        obs_list = self.env.unwrapped.observation()
        uids = [o['active'] for o in obs_list]

        if self.unit_based_obs:
            _compute_relative_pos(obs_list, uids[:self.number_of_left_players_agent_controls], uids[self.number_of_left_players_agent_controls:])
            print('get ball info')
            _get_ball_info(self.env.unwrapped.observation(), self.number_of_left_players_agent_controls, self.number_of_right_players_agent_controls)

            obs_list = self.env.unwrapped.observation()
            _check_obs_consistency(obs, obs_list)
            agent_obs = _compute_all_obs(
                obs_list, 
                self.number_of_left_players_agent_controls, 
                self.number_of_right_players_agent_controls
            )
            assert len(agent_obs) == len(action) == len(reward) == self.n_agents, \
                (len(agent_obs), len(action), len(reward))
            for obs, a, r in zip(agent_obs, action, reward):
                obs['prev_action'] = a
                obs['prev_reward'] = r
        else:
            if self.n_units == 1:
                obs = np.expand_dims(obs, 0)

            agent_obs = [dict(
                obs=obs[uids], 
                global_state=obs[uids], 
                prev_reward=reward[aid], 
                prev_action=action[aid], 
                action_mask=np.ones((len(uids), self.action_dim[aid]), bool) 
            ) for aid, uids in enumerate(self.aid2uids)]

        if self.add_role_to_obs:
            roles = _get_roles(obs_list, uids)
            for obs, uids in zip(agent_obs, self.aid2uids):
                obs['role'] = roles[uids]

        return agent_obs

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', '-l', type=int, default=1)
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
        'unit_based_obs': False, 
        'infield_penalty': 0, 
        'add_role_to_obs': True,
        'uid2aid': None
    }

    from utility.display import print_dict_info, print_dict
    env = GRF(**config)
    print(env.obs_shape)
    env.reset()
    for i in range(args.step):
        a = env.random_action()
        o, r, d, info = env.step(a)
        pr = o[-1]['prev_reward']
        if np.any(r[-1] != 0) or np.any(pr != 0):
            print(i, 'previous reward', r, o[-1]['prev_reward'])

    # for oo in o:
    #     print_dict(oo)
    print(env._score)
    for dd in d:
        print(dd)
