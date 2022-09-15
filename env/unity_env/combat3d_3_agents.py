import math
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cloudpickle
import functools
from typing import List
from core.elements.builder import ElementsBuilder
from env.typing import EnvOutput
import numpy as np
import gym
from interface import UnityInterface
from run.utils import search_for_config
from core.tf_config import configure_gpu

"""  Naming Conventions:
An Agent is the one that makes decisions, it's typically a RL agent
A Unit is a controllable unit in the environment.
Each agent may control a number of units.
All ids start from 0 and increases sequentially
"""

MAX_LEN = 50
SHOT_DIS = 40


def compute_aid2uids(uid2aid):
    """ Compute aid2uids from uid2aid """
    aid2uids = []
    for uid, aid in enumerate(uid2aid):
        if aid > len(aid2uids):
            raise ValueError(f'uid2aid({uid2aid}) is not sorted in order')
        if aid == len(aid2uids):
            aid2uids.append((uid,))
        else:
            aid2uids[aid] += (uid,)

    aid2uids = [np.array(uids, np.int32) for uids in aid2uids]

    return aid2uids


# NOTE: Keep the class name fixed; do not invent a new one!
# We do not rely on this for distinction!
class UnityEnv:
    def __init__(
            self,
            uid2aid,
            n_envs,
            unity_config,
            seed=None,
            frame_skip=50,
            is_action_discrete=True,
            reward_config={
                'detect_reward': 0,
                'main_dead_reward': 0,
                'blue_dead_reward': 1,
                'grid_reward': 0
            },
            **kwargs
    ):
        configure_gpu(None)
        self.uid2aid: list = uid2aid
        self.aid2uids = compute_aid2uids(self.uid2aid)
        self.n_agents = len(self.aid2uids)
        self.n_units = len(self.uid2aid)
        self.frame_skip = 50
        self.n_red_main = 1
        self.n_red_allies = 4
        self.n_blue_main = 2
        self._seed = np.random.randint(1000) if seed is None else seed
        self.n_planes = self.n_red_main + self.n_red_allies + self.n_blue_main
        self.n_envs = n_envs
        self.unity_config = unity_config
        self.max_episode_steps = kwargs['max_episode_steps']
        self.reward_config = reward_config
        self.use_action_mask = False  # if action mask is used
        self.use_life_mask = False  # if life mask is used
        self.unity_config['file_name'] = None
        self.unity_config['worker_id'] = 0
        self.action_dim = [4, 3, 4]
        self.action_space = [gym.spaces.Box(low=-1, high=1, shape=(ad,)) for ad in self.action_dim]
        self._obs_dim = [196, 112, 107]
        self._global_state_dim = [196, 112, 107]

        self.obs_shape = [dict(
            obs=self._get_obs_shape(aid),
            global_state=self._get_global_state_shape(aid),
            prev_reward=(),
            prev_action=(self.action_dim[aid],),
        ) for aid in range(self.n_agents)]
        self.obs_dtype = [dict(
            obs=np.float32,
            global_state=np.float32,
            prev_reward=np.float32,
            prev_action=np.float32,
        ) for _ in range(self.n_agents)]
        if self.use_life_mask:
            for aid in range(self.n_agents):
                # 1 if the unit is alive, otherwise 0
                self.obs_shape[aid]['life_mask'] = ()
                self.obs_dtype[aid]['life_mask'] = np.float32
        if self.use_action_mask:
            for aid in range(self.n_agents):
                # 1 if the action is valid, otherwise 0
                if self.is_action_discrete:
                    self.obs_shape[aid]['action_mask'] = (self.action_space[aid].n,)
                else:
                    self.obs_shape[aid]['action_mask'] = self.action_space[aid].shape
                self.obs_dtype[aid]['action_mask'] = bool

        self._win_rate = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._lose_rate = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._draw_rate = np.zeros((self.n_envs, self.n_units), dtype=np.float32)

        # 不同的score，分别对应击毁蓝方、自己死亡、检测到蓝方的reward
        self._dense_score = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._dense_penalty = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)

        # The length of the episode
        self._epslen = np.zeros(self.n_envs, np.int32)

        # 红蓝双方存活状况
        self.alive_units = np.ones((self.n_envs, self.n_planes), np.int32)

        self.pre_reward = np.zeros((self.n_envs, self.n_units), np.float32)
        self.prev_action = None
        self._consecutive_action = np.zeros((self.n_envs, self.n_units), bool)
        self.fly_red = None
        self.fly_blue = None
        self.fly_control_steps = 1
        self.total_steps = 0
        self.shot_steps = 0
        self.unity_shot_steps = 0
        self.total_step_time = 0
        self.env_step_time = 0
        self.locked_warning = np.zeros((self.n_envs, self.n_planes), bool)

    def random_action(self):
        actions = []
        for aid, uids in enumerate(self.aid2uids):
            a = np.array([[self.action_space[aid].sample() for _ in uids] for _ in range(self.n_envs)], np.float32)
            actions.append(a)
        return actions

    def reset(self):
        self.env = UnityInterface(**self.unity_config)
        self._red_names, self._blue_names = self._get_names(self.env)
        self.blue_radar_action = -1 * np.ones((2, 1))
        self.blue_shot_action = np.zeros((2, 1))
        self._win_rate = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._lose_rate = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._draw_rate = np.zeros((self.n_envs, self.n_units), dtype=np.float32)

        self._dense_score = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._dense_penalty = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)

        # self._win_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        # self._lose_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        # self._detect_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        # self._missile_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)

        self._epslen = np.zeros(self.n_envs, np.int32)
        self.red_shot_time = 0
        self.blue_shot_time = np.zeros(2)
        self.alive_units = np.ones((self.n_envs, self.n_planes), np.int32)
        self.detect_units = np.zeros((self.n_envs, self.n_units), bool)
        self.enemy_position = np.zeros((self.n_envs, self.n_blue_main, 4))
        self.pre_reward = np.zeros((self.n_envs, self.n_units), np.float32)
        self._consecutive_action = np.zeros((self.n_envs, self.n_units), bool)
        self.prev_action = None
        self.fly_red = self.init_red_fly_model()
        self.dis = np.zeros((7, 3))
        decision_steps, terminal_steps = self.env.reset()
        self.decision_steps = decision_steps
        return self._get_obs(decision_steps)

    def init_red_fly_model(self):
        directory = ["../../logs/unity-combat2d/sync-hm/0624-fixed-drag-n_runners=30-entropy_coef=5e-4/seed=None/a0/i1-v1"]
        configs = [search_for_config(d) for d in directory]
        config = configs[0]
        env_stats = {'obs_shape': [{'obs': (14,), 'global_state': (14,), 'prev_reward': (), 'prev_action': (4,)}],
                     'obs_dtype': [{'obs': np.float32, 'global_state': np.float32, 'prev_reward': np.float32,
                                    'prev_action': np.float32}],
                     'action_shape': [(4,)], 'action_dim': [4], 'action_low': None, 'action_high': None,
                     'is_action_discrete': [False], 'action_dtype': [np.float32], 'n_agents': 1, 'n_units': 1,
                     'uid2aid': [0], 'aid2uids': [np.array([0])], 'use_life_mask': False, 'use_action_mask': False,
                     'is_multi_agent': True, 'n_workers': 1, 'n_envs': 1}
        self.last_angle = np.zeros(3)
        self.fly_obs_shape = [dict(
            obs=(14,),
            global_state=(14,),
            prev_reward=(),
            prev_action=(4,),
        )]
        self.fly_obs_dtype = [dict(
            obs=np.float32,
            global_state=np.float32,
            prev_reward=np.float32,
            prev_action=np.float32,
        )]

        builder = ElementsBuilder(config, env_stats, config.algorithm)
        elements = builder.build_acting_agent_from_scratch(to_build_for_eval=True)
        agent = elements.agent
        path = "../../logs/unity-combat2d/sync-hm/0624-fixed-drag-n_runners=30-entropy_coef=5e-4/seed=None/a0/i1-v1/params.pkl"

        with open(path, 'rb') as f:
            weights = cloudpickle.load(f)
            agent.set_weights(weights)
        return agent

    def get_fly_obs(self, ds, dis):
        obs = [{}]
        all_states = ds
        alive = all_states[1]
        vel = all_states[5:8]
        v_scalar = np.linalg.norm(vel)
        angle = all_states[8:11]
        for i in range(3):
            while abs(angle[i]) > 180:
                angle[i] = angle[i] - math.copysign(360, angle[i])

        theta = angle[1] / 180
        phi = angle[0] / 180
        v = np.array([v_scalar, theta, phi])
        posture = np.concatenate((np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))))
        height = all_states[3]

        overload = all_states[14]
        oil = all_states[15]

        delta_roll = (angle[2] - self.last_angle[2]) / 180
        if abs(delta_roll) > 1:
            delta_roll = delta_roll - math.copysign(2, delta_roll)

        roll_v = delta_roll

        overload = np.clip(overload, -2, 2)

        one_obs = np.hstack((
            v,
            roll_v,
            posture,
            height / 20,
            dis,
            # self.overload / 2,
            # oil
        ))
        self.last_angle = angle.copy()
        # print('obs:' + str(one_obs))
        observations = {}
        observations['obs'] = one_obs
        observations['global_state'] = one_obs
        observations['life_mask'] = [1]
        mask = [1, 1, 1]
        observations['action_mask'] = mask
        observations['prev_reward'] = np.zeros(1, np.float32)
        observations['prev_action'] = np.zeros(4, np.float32)
        all_obs = [observations]

        for aid in range(1):
            for k in self.fly_obs_shape[aid].keys():
                obs[aid][k] = np.zeros(
                    (1, 1, *self.fly_obs_shape[aid][k]),
                    dtype=self.fly_obs_dtype[aid][k]
                )
                obs[aid][k][0] = all_obs[0][k]

        return obs

    def step(self, actions):
        #actions[0][0][0] = np.zeros(4)
        start = time.time()
        # while self._epslen < 80:
        #     for i in range(self.frame_skip):
        #         reset, decision_steps, terminal_steps = self.env.step()
        #         if reset:
        #             break
        #     self._epslen += 1
        self.total_steps += 1
        if actions[0][0][0][3] > 0.5:
            self.shot_steps += 1
        for i in range(self.fly_control_steps):
            fly_act = {}
            for id, name in enumerate(self._red_names[0] + self._blue_names[0]):
                self.dis[id] = np.zeros(3)
                if id == 0:
                    self.dis[id][0] = actions[0][0][0][0]/10
                    self.dis[id][1] = actions[0][0][0][1]
                    self.dis[id][2] = actions[0][0][0][2]/12
                if 0 < id < 5:
                    self.dis[id][0] = actions[1][0][id-1][0]/10
                    self.dis[id][1] = actions[1][0][id-1][1]
                    self.dis[id][2] = actions[1][0][id-1][2]/12
                if id >= 5:
                    self.dis[id][0] = actions[2][0][id - 5][0] / 10
                    self.dis[id][1] = actions[2][0][id - 5][1]
                    self.dis[id][2] = actions[2][0][id - 5][2] / 12

                fly_obs = self.get_fly_obs(self.decision_steps[name].obs[0][0], self.dis[id])

                # batch_obs = {}
                # for k, v in fly_obs[0].items():
                #     batch_obs[k] = np.expand_dims(v, 0)
                reward = np.zeros((1, 1))
                discount = np.zeros((1, 1))
                reset = np.zeros((1, 1))
                env_output = EnvOutput(fly_obs[0], reward, discount, reset)
                if abs(self.dis[id][0]) < 0.01 and abs(self.dis[id][1]) < 0.02 and abs(self.dis[id][2]) < 0.02:
                    fly_act[name] = (np.zeros((1, 1, 4)), {})
                else:
                    fly_act[name] = self.fly_red(
                    env_output,
                    evaluation=True)

            for j in range(10):
                if i == 0 and j == 0:
                    self.red_radar_action = {}
                    self.red_shot_action = {}
                    for uid, n in enumerate(self._red_names[0]):
                        self.red_radar_action[n] = 1 if self.alive_units[0][-2] == 1 else 2
                        if uid == 0:
                            # self.red_shot_action[n] = 1
                            self.red_shot_action[n] = actions[0][0][0][3] if self.red_missile_left[0][0][0] > 0 else 0
                            if self.red_shot_action[n] > 0.5 and self._epslen[0] - self.red_shot_time > 20:
                                self.red_shot_action[n] = 1
                                self.red_shot_time = self._epslen[0]
                            else:
                                self.red_shot_action[n] = 0
                        else:
                            self.red_shot_action[n] = 0

                    blue_radar_action = {}
                    blue_shot_action = {}
                    for uid, n in enumerate(self._blue_names[0]):
                        blue_radar_action[n] = 1
                        blue_shot_action[n] = self.blue_shot_action[uid][0] if self.blue_missile_left[0][uid][0] > 0 else 0
                        if blue_shot_action[n] > 0.5 and self._epslen[0] - self.blue_shot_time[uid] > 20:
                            blue_shot_action[n] = 1
                            self.blue_shot_time[uid] = self._epslen[0]
                        else:
                            blue_shot_action[n] = 0

                    self._set_low_action(self._red_names, fly_act.copy(), self.red_radar_action.copy(),
                                         self.red_shot_action.copy())
                    self._set_low_action(self._blue_names, fly_act.copy(), blue_radar_action.copy(),
                                         blue_shot_action.copy())
                else:
                    self.red_radar_action = {}
                    self.red_shot_action = {}
                    for uid, n in enumerate(self._red_names[0]):
                        self.red_radar_action[n] = 1
                        self.red_shot_action[n] = 0
                    blue_radar_action = {}
                    blue_shot_action = {}
                    for uid, n in enumerate(self._blue_names[0]):
                        blue_radar_action[n] = 1
                        blue_shot_action[n] = 0
                    self._set_low_action(self._red_names, fly_act.copy(), self.red_radar_action.copy(),
                                         self.red_shot_action.copy())
                    self._set_low_action(self._blue_names, fly_act.copy(), blue_radar_action.copy(),
                                         blue_shot_action.copy())
                start1 = time.time()
                reset, decision_steps, terminal_steps = self.env.step()
                end1 = time.time()
                self.env_step_time += end1 - start1
                self.decision_steps = decision_steps
                if np.count_nonzero(decision_steps['E0_Red_0?team=0'].obs[0][0]) == 0:
                    break
                if np.count_nonzero(decision_steps['E0_Blue_0?team=0'].obs[0][0]) == 0 \
                        and np.count_nonzero(decision_steps['E0_Blue_1?team=0'].obs[0][0]) == 0:
                    break

        self._epslen += 1

        agent_obs = self._get_obs(self.decision_steps, self.pre_reward)

        done, reward, penalty = self._get_done_and_reward()
        if np.count_nonzero(self.decision_steps['E0_Red_0?team=0'].obs[0][0]) == 0:
            done[0] = True
        if np.count_nonzero(self.decision_steps['E0_Blue_0?team=0'].obs[0][0]) == 0 \
                and np.count_nonzero(self.decision_steps['E0_Blue_1?team=0'].obs[0][0]) == 0:
            done[0] = True

        for i in range(self.n_envs):
            if self._epslen[i] > self.max_episode_steps:
                done[i] = True
                self._draw_rate[i] += 1

        discount = 1 - done  # we return discount instead of done

        assert reward.shape == (self.n_envs, self.n_planes), reward.shape
        assert discount.shape == (self.n_envs,), discount.shape
        # obtain ndarrays of shape (n_envs, n_units)
        # rewards = np.tile(reward.reshape(-1, 1), (1, self.n_units))
        rewards = reward
        discounts = np.tile(discount.reshape(-1, 1), (1, self.n_planes))

        self._dense_score += rewards
        self._dense_penalty += penalty

        reset = done
        assert reset.shape == (self.n_envs,), reset.shape
        resets = np.tile(reset.reshape(-1, 1), (1, self.n_units))

        self.prev_action = actions

        alive_blue = np.zeros(self.n_envs)
        alive_main = np.zeros(self.n_envs)
        alive_ally = np.zeros(self.n_envs)
        alive_blue[0] = 2 - ((self.alive_units[0][-2] != 1) + (self.alive_units[0][-1] != 1))
        alive_main[0] = 1 - (self.alive_units[0][0] != 1)
        alive_ally[0] = 4 - ((self.alive_units[0][1] != 1) +
                             (self.alive_units[0][2] != 1) +
                             (self.alive_units[0][3] != 1) +
                             (self.alive_units[0][4] != 1)
                             )
        self._info = [dict(
            score=self._win_rate[i].copy(),
            win_rate=self._win_rate[i].copy(),
            lose_rate=self._lose_rate[i].copy(),
            draw_rate=self._draw_rate[i].copy(),
            dense_score=self._dense_score[i].copy(),
            dense_penalty=self._dense_penalty[i].copy(),
            epslen=np.array([self._epslen[i]] * self.n_units),
            game_over=np.array([discount[i] == 0] * self.n_units),
            left_missile_red=np.hstack([self.red_missile_left[i][0][0]] * self.n_units),
            left_missile_blue0=np.array([self.blue_missile_left[i][0][0]] * self.n_units),
            left_missile_blue1=np.array([self.blue_missile_left[i][1][0]] * self.n_units),
            alive_blue=np.array([alive_blue[i]] * self.n_units),
            alive_main=np.array([alive_main[i]] * self.n_units),
            alive_ally=np.array([alive_ally[i]] * self.n_units),
            total_steps=np.array([self.total_steps] * self.n_units),
            shot_steps=np.array([self.shot_steps] * self.n_units),
            unity_shot_steps=np.array([self.unity_shot_steps] * self.n_units),
            total_step=np.array([self.total_step_time] * self.n_units),
            total_env_step=np.array([self.env_step_time] * self.n_units),
        ) for i in range(self.n_envs)]
        agent_reward = [rewards[:, uids] for uids in self.aid2uids]
        agent_discount = [discounts[:, uids] for uids in self.aid2uids]
        agent_reset = [resets[:, uids] for uids in self.aid2uids]

        reset_env = []
        for i in range(self.n_envs):
            if done[i]:
                reset_env.append(i + 1)
                self._dense_score[i] = np.zeros(self.n_units, np.float32)
                self._dense_penalty[i] = np.zeros(self.n_units, np.float32)
                self._win_rate[i] = np.zeros(self.n_units, np.float32)
                self._lose_rate[i] = np.zeros(self.n_units, np.float32)
                self._draw_rate[i] = np.zeros(self.n_units, np.float32)
                self.alive_units[i] = np.ones(self.n_planes, np.int32)
                self.locked_warning = np.zeros((self.n_envs, 7), bool)
                self._epslen[i] = 0
                self.red_shot_time = 0
                self.blue_shot_time = np.zeros(2)
                self.prev_action = None

        self.env.reset_envs_with_ids(reset_env) if len(reset_env) != 0 else None
        end = time.time()
        self.total_step_time += (end - start)
        return agent_obs, agent_reward, agent_discount, agent_reset

    def info(self):
        return self._info

    def close(self):
        # close the environment
        pass

    def _get_obs_shape(self, aid):
        return (self._obs_dim[aid],)

    def seed(self, seed):
        """Returns the random seed used by the environment."""
        self._seed = seed

    def _get_global_state_shape(self, aid):
        return (self._global_state_dim[aid],)

    def _get_names(self, env):
        teams = {}
        for name in env.get_behavior_names():
            team = name[name.index('team'):]
            if teams.__contains__(team) is False:
                teams[team] = [name]
            else:
                teams[team].append(name)

        red_names = []
        blue_names = []

        def team_cmp(t1, t2):
            k1 = str(t1[0])
            k2 = str(t2[0])
            team1 = int(k1[k1.index('=') + 1:])
            team2 = int(k2[k2.index('=') + 1:])
            if team1 < team2:
                return -1
            if team1 == team2:
                return 0
            if team1 > team2:
                return 1

        for k, v in sorted(teams.items(), key=functools.cmp_to_key(team_cmp)):
            red_names.append([name for name in sorted(v) if name.__contains__('Red')])
            blue_names.append([name for name in sorted(v) if name.__contains__('Blue')])

        return red_names, blue_names

    def _get_obs(self, decision_step, reward=None, action=None):
        def extract_obs(names):
            team_obs = [
                [decision_step[name].obs for name in team]
                for team in names
            ]
            return team_obs

        obs = [{} for _ in range(self.n_agents)]

        red_obs = extract_obs(self._red_names)
        blue_obs = extract_obs(self._blue_names)

        all_obs = self._process_obs(red_obs, blue_obs, reward, action)

        for aid in range(self.n_agents):
            for k in self.obs_shape[aid].keys():
                obs[aid][k] = np.zeros(
                    (self.n_envs, len(self.aid2uids[aid]), *self.obs_shape[aid][k]),
                    dtype=self.obs_dtype[aid][k]
                )
                for n in range(self.n_envs):
                    for id, uid in enumerate(self.aid2uids[aid]):
                        obs[aid][k][n][id] = all_obs[n][k][uid]
        return obs

    def _process_obs(self, red_info, blue_info, reward, action):
        red_observation = []
        self.blue_missile_left = np.zeros((self.n_envs, self.n_blue_main, 3), np.int32)
        self.red_missile_left = np.zeros((self.n_envs, self.n_red_main, 3), np.int32)

        # missile_end记录蛋是否已爆
        self.missile_end = np.zeros((self.n_envs, self.n_blue_main + self.n_red_main))
        self.detect_units = np.zeros((self.n_envs, self.n_units), bool)

        for team in range(self.n_envs):
            # 获取红方主僚机的射线信息、状态信息、导弹信息 red main（21+88+12） red ally（21+12） blue（21+88+24）
            red_main_info = np.array([i[0] for i in red_info[team][:self.n_red_main]]).squeeze(1)
            red_ally_info = np.array([i[0] for i in red_info[team][self.n_red_main:]]).squeeze(1)

            red_main_self = red_main_info[:, 0:21]
            red_main_weapon = red_main_info[:, 21:109]
            red_main_radar = red_main_info[:, 109:]

            red_ally_self = red_ally_info[:, 0:21]
            red_ally_radar = red_ally_info[:, 21:]

            red_self = np.concatenate((red_main_self, red_ally_self), axis=0)
            red_radar = np.concatenate((red_main_radar, red_ally_radar), axis=0)

            # 获取蓝方射线信息、状态信息、导弹信息
            blue_all_info = np.array([i[0] for i in blue_info[team]]).squeeze(1)
            blue_self = blue_all_info[:, 0:21]

            blue_weapon = blue_all_info[:, 21:109]
            blue_radar = blue_all_info[:, 109:]

            # 根据状态信息获取获取各个飞机相互之间的相对坐标、距离、角色等
            all_self = np.concatenate((red_self, blue_self), axis=0)
            positions = self._get_all_positions(all_self)
            rel_pos = self._get_rel_positions(positions)
            plane_infos = self._get_all_plane_info(team, all_self)

            assert rel_pos.shape == (self.n_planes, self.n_planes, 3), rel_pos.shape
            assert len(plane_infos) == self.n_planes, len(plane_infos)

            self.red_missile_left[team] = red_main_weapon[:, 1:4]
            self.blue_missile_left[team] = blue_weapon[:, 1:4]

            observations = {}
            all_obs, all_global_state, all_alive, all_action_mask, all_prev_action = [], [], [], [], []

            # 遍历红方飞机，计算observation
            for i, name in enumerate(self._red_names[team]):
                obs = []
                alive = red_self[i][1]
                self.alive_units[team][i] = alive if self.alive_units[team][i] != -1 else -1
                vel = red_self[i][5:8]
                angle = red_self[i][11:14]
                for tt in range(3):
                    while abs(angle[tt]) > 180:
                        angle[tt] = angle[tt] - math.copysign(360, angle[tt])
                overload = red_self[i][17]
                oil = red_self[i][18]
                locked_warning = red_self[i][19]
                shot_warning = red_self[i][20]
                self.locked_warning[team][i] = locked_warning
                # me_missile = self._get_target_missile(blue_weapon, team, i, positions[i])
                self_cont = [vel,
                             np.array([overload]),
                             np.concatenate([np.sin(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]),
                             np.array([oil])
                             ]
                obs.append(np.concatenate(self_cont))
                threshold = 1000
                oil_warning = 1 if oil < threshold else 0
                which_missile = red_main_weapon[i][0] if i < self.n_red_main else 0
                mid_dis_missile = red_main_weapon[i][1] if i < self.n_red_main else 0
                short_dis_missile = red_main_weapon[i][2] if i < self.n_red_main else 0
                disrupt_missile = red_main_weapon[i][3] if i < self.n_red_main else 0
                alive_blue_num = sum(blue_self[:, 1])

                self_disc = np.array([oil_warning,
                                      locked_warning,
                                      shot_warning,
                                      mid_dis_missile,
                                      short_dis_missile,
                                      disrupt_missile,
                                      which_missile,
                                      alive_blue_num])
                obs.append(self_disc)

                if i < self.n_red_main:
                    self_miss_info = []
                    self_miss_state = []
                    for index in range(4, len(red_main_weapon[i]), 14):
                        miss_type = np.array([red_main_weapon[i][index]])
                        miss_pos = red_main_weapon[i][index + 1:index + 4]
                        miss_vel = red_main_weapon[i][index + 4:index + 7]
                        miss_angle = red_main_weapon[i][index + 7:index + 10]

                        miss_direction = red_main_weapon[i][index + 10:index + 13]
                        miss_state = np.array([red_main_weapon[i][index + 13]])
                        self_miss_info.append(miss_type)
                        self_miss_info.append(miss_pos)
                        self_miss_info.append(miss_vel)
                        self_miss_info.append(
                            np.concatenate([np.sin(np.deg2rad(miss_angle)), np.sin(np.deg2rad(miss_angle))]))
                        self_miss_state.append(miss_state)
                    missile_cont = np.hstack(self_miss_info).flatten()
                    missile_disc = np.hstack(self_miss_state).flatten()
                    obs.append(missile_cont)
                    obs.append(missile_disc)
                #else:
                #    obs.append(np.zeros(84))
                radar_state = red_radar[i][0]
                radar_target = red_radar[i][1]
                radar_angle = red_radar[i][2]
                radar_distance = red_radar[i][3]

                radar_cont = np.array([radar_angle, radar_distance])
                radar_disc = np.array([radar_state, radar_target])
                obs.append(radar_cont)
                obs.append(radar_disc)

                except_me_info = [info for j, info in enumerate(plane_infos) if j != i]
                except_me_ally_info = except_me_info[0:-2]
                except_me_enemy_info = except_me_info[-2:]

                except_me_rel_pos = rel_pos[i]
                except_me_rel_pos = np.delete(except_me_rel_pos, i, 0)
                except_me_ally_rel_pos = except_me_rel_pos[0:4]
                except_me_enemy_rel_pos = except_me_rel_pos[-2:]
                ally_dis = [np.linalg.norm(k) for k in except_me_ally_rel_pos]
                ally_cont = [except_me_ally_rel_pos.flatten(),
                             np.concatenate([np.concatenate([np.sin(np.deg2rad(k[6:9])), np.cos(np.deg2rad(k[6:9]))])
                                             for k in except_me_ally_info]).reshape((4, 6)).flatten()
                             ]

                ally_dist = except_me_ally_info[0:3]
                obs.append(np.concatenate(ally_cont))
                obs.append(np.concatenate(ally_dist))

                # other_state_info = copy.deepcopy(plane_infos)
                # other_position = [rp for j, rp in enumerate(rel_pos[i]) if j != i]
                # for j, blue_info in enumerate(other_position[-self.n_blue_main:]):
                #    blue_info = self.enemy_position[team, j, :2]  # 用env维护的蓝方位置信息替换真实位置信息
                # other_distance = np.array([np.linalg.norm(j) for j in other_position])
                enemy_dis = np.array([np.linalg.norm(except_me_enemy_rel_pos[k]) for k in range(2)])

                enemy_cont = [
                    except_me_rel_pos[-2:].flatten(),
                    np.concatenate(
                        [np.concatenate([np.sin(np.deg2rad(k[6:9])),
                                         np.cos(np.deg2rad(k[6:9]))]) for k in except_me_enemy_info]).flatten()
                ]

                obs.append(enemy_dis)
                obs.append(np.concatenate(enemy_cont))

                # 获取是否探测到蓝方
                # if i < self.n_red_main:
                #    n_enemies = self._get_detect_blues(red_main_ray_info[i], i)
                # else:
                #    n_enemies = self._get_detect_blues(red_ally_ray_info[i - self.n_red_main], i)

                # rewards[i][v] += sum(detect_flag) * self.reward_setting['DETECT_REWARD']
                # detect_flag = np.zeros(self.n_blue_main, np.float32)
                # detect_flag[0:n_enemies - 1] = 1 if n_enemies != 0 else 0

                # self.detect_units[team][i] += n_enemies
                # self.detect_steps[team][i] += sum(detect_flag)
                # self.alive_steps[team][i] += alive == 1

                # for bid in range(2):
                #     self.enemy_position[team][bid] = blue_state_info[bid, 2:6]
                # 蓝方位置信息
                # for bid in range(2):
                #     if detect_flag[bid] == 1:
                #         self.enemy_position[team][bid] = blue_state_info[bid, 2:6]
                #     else:
                #         # TODO: maintain an additional counter
                #         if self._epslen[team] % 7 == 0:
                #             self.enemy_position[team][bid] = blue_state_info[bid, 2:6] + np.random.randn(4)
                #         else:
                #             self.enemy_position[team][bid][0:2] = self._predict_position(
                #                 self.enemy_position[team][bid][0:2],
                #                 self.enemy_position[team][bid][2:4],
                #                 self._epslen[team] % 7)
                # 锁定主机和自己的最近的蛋
                # if me_missile != 0:
                #     assert me_missile < MAX_LEN, me_missile
                #     self.missile_dis[team][v] -= me_missile
                blue_missile_left = blue_weapon[:, 1]

                all_action_mask.append(np.ones(self.action_dim[self.uid2aid[i]], bool))

                obs = np.concatenate(obs)
                global_state = obs
                if not alive:
                    obs = np.zeros_like(obs)
                all_obs.append(obs)

                all_global_state.append(global_state)
                all_alive.append(alive)
                prev_action = np.zeros((self.action_dim[self.uid2aid[i]]), np.float32)
                all_prev_action.append(prev_action)

            for i, name in enumerate(self._blue_names[team]):
                obs = []
                alive = blue_self[i][1]
                self.alive_units[team][i+5] = alive if self.alive_units[team][i+5] != -1 else -1
                vel = blue_self[i][5:8]
                angle = blue_self[i][11:14]
                for tt in range(3):
                    while abs(angle[tt]) > 180:
                        angle[tt] = angle[tt] - math.copysign(360, angle[tt])
                overload = blue_self[i][17]
                oil = blue_self[i][18]
                locked_warning = blue_self[i][19]
                shot_warning = blue_self[i][20]
                self.locked_warning[team][i+5] = locked_warning
                # me_missile = self._get_target_missile(blue_weapon, team, i, positions[i])
                self_cont = [vel,
                             np.array([overload]),
                             np.concatenate([np.sin(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]),
                             np.array([oil])
                             ]
                obs.append(np.concatenate(self_cont))
                threshold = 1000
                oil_warning = 1 if oil < threshold else 0
                which_missile = blue_weapon[i][0]
                mid_dis_missile = blue_weapon[i][1]
                short_dis_missile = blue_weapon[i][2]
                disrupt_missile = blue_weapon[i][3]
                alive_red_num = sum(red_self[:, 1])

                self_disc = np.array([oil_warning,
                                      locked_warning,
                                      shot_warning,
                                      mid_dis_missile,
                                      short_dis_missile,
                                      disrupt_missile,
                                      which_missile,
                                      alive_red_num])
                obs.append(self_disc)

                self_miss_info = []
                self_miss_state = []
                for index in range(4, len(blue_weapon[i]), 14):
                    miss_type = np.array([blue_weapon[i][index]])
                    miss_pos = blue_weapon[i][index + 1:index + 4]
                    miss_vel = blue_weapon[i][index + 4:index + 7]
                    miss_angle = blue_weapon[i][index + 7:index + 10]

                    miss_direction = blue_weapon[i][index + 10:index + 13]
                    miss_state = np.array([blue_weapon[i][index + 13]])
                    self_miss_info.append(miss_type)
                    self_miss_info.append(miss_pos)
                    self_miss_info.append(miss_vel)
                    self_miss_info.append(
                        np.concatenate([np.sin(np.deg2rad(miss_angle)), np.sin(np.deg2rad(miss_angle))]))
                    self_miss_state.append(miss_state)
                missile_cont = np.hstack(self_miss_info).flatten()
                missile_disc = np.hstack(self_miss_state).flatten()
                obs.append(missile_cont)
                obs.append(missile_disc)
                #else:
                #    obs.append(np.zeros(84))
                radar_state = blue_radar[i][0]
                radar_target = blue_radar[i][1]
                radar_angle = blue_radar[i][2]
                radar_distance = blue_radar[i][3]

                radar_cont = np.array([radar_angle, radar_distance])
                radar_disc = np.array([radar_state, radar_target])
                obs.append(radar_cont)
                obs.append(radar_disc)
    
                all_action_mask.append(np.ones(self.action_dim[self.uid2aid[i+5]], bool))

                obs = np.concatenate(obs)
                global_state = obs
                if not alive:
                    obs = np.zeros_like(obs)
                all_obs.append(obs)

                all_global_state.append(global_state)
                all_alive.append(alive)
                prev_action = np.zeros((self.action_dim[self.uid2aid[i+5]]), np.float32)
                all_prev_action.append(prev_action)

            observations['obs'] = all_obs
            observations['global_state'] = all_global_state
            observations['life_mask'] = all_alive
            observations['action_mask'] = all_action_mask
            observations['prev_reward'] = np.zeros(self.n_units, np.float32) if reward is None else reward[team]
            observations['prev_action'] = all_prev_action

            red_observation.append(observations)

        return red_observation

    def _set_blue_action(self, radar_action, shot_action):
        for name in self._blue_names[0]:
            action = np.zeros(4)
            action_tuple = self.env.get_action_tuple()
            action = np.hstack((action.reshape((1, 4)), np.zeros((1, 2))))
            action_tuple.add_continuous(action)

            if name.startswith('E0_Blue'):
                l = np.zeros((1, 4), np.int32)
                l[0][0] = shot_action[0]
                l[0][2] = radar_action[0] + 1
                action_tuple.add_discrete(l)
            else:
                l = np.zeros((1, 4), np.int32)
                l[0][0] = shot_action[0]
                l[0][2] = radar_action[0] + 1
                action_tuple.add_discrete(l)
            self.env.set_actions(name, action_tuple)

    def _update_missile_life(self, blue_missile_info, team):
        for id, missile_info in enumerate(blue_missile_info):
            for missle_left in range(2, len(missile_info) - 5 - 9 + 1, 9):
                if missile_info[missle_left + 8] == 2:
                    self.missile_end[team][id + 1] += 1

    def _update_blue_alive(self, team, blue_state_info):
        for id, blue in enumerate(self._blue_names[team]):
            # if blue_state_info[id][1] == 1:
            #    self.alive_steps[team][id + 5] += 1
            if blue_state_info[id][1] == 0 and self.alive_units[team][id + 5] == 1:
                self.alive_units[team][id + 5] = 0

    def _get_all_positions(self, all_states):
        pos = np.array(all_states[:, 2:5])
        return pos

    def _get_rel_positions(self, pos):
        rel_pos = np.expand_dims(pos, 0) - np.expand_dims(pos, 1)
        return rel_pos

    def _get_all_plane_info(self, team, all_states):
        ret = []
        for i, name in enumerate(self._red_names[team] + self._blue_names[team]):
            state = all_states[i]
            role = self._get_plane_role(name)
            velocity = list(state[5:8])
            direction = list(state[8:11])
            alive = [1, 0] if state[1] == 1 else [0, 1]

            ret.append(role + velocity + direction + alive)
            # ret.append(role + alive)
        return ret

    def _get_plane_role(self, name):
        if name.startswith('E0_Red_0'):
            return [1, 0, 0]
        else:
            if name.startswith('E0_Red'):
                return [0, 1, 0]
            if name.startswith('E0_Blue'):
                return [0, 0, 1]

    def _set_low_action(self, names, move_actions, radar_actions, shot_actions):
        def to_unity_action(name, move_action, radar_action, shot_action):
            # cont_a = np.hstack((np.ones((1,4)), np.zeros((1, 2))))
            cont_a = np.hstack((move_action[0:4].reshape((1, 4)), np.zeros((1, 2))))
            if name.startswith('E0_Red_0'):
                dis_a = np.zeros((1, 4), np.int32)
                if shot_action < 0.5:
                    dis_a[0][0] = 0
                else:
                    dis_a[0][0] = 1
                    self.unity_shot_steps += 1
                dis_a[0][2] = radar_action
            else:
                if name.startswith('E0_Red'):
                    dis_a = np.zeros((1, 2), np.int32)
            if name.startswith('E0_Blue'):
                dis_a = np.zeros((1, 4), np.int32)
                dis_a[0][0] = 0 if shot_action < 0.5 else 1
                dis_a[0][2] = radar_action

            return cont_a, dis_a

        for uid, n in enumerate(names[0]):
            move_action = move_actions[n][0][0][0]
            move_action = np.clip(move_action, -1, 1)

            radar_action = radar_actions[n]
            shot_action = shot_actions[n]
            cont_action, dist_action = to_unity_action(n, move_action, radar_action, shot_action)
            action_tuple = self.env.get_action_tuple()
            action_tuple.add_discrete(dist_action)
            action_tuple.add_continuous(cont_action)
            self.env.set_actions(n, action_tuple)

    def _set_action(self, names: List[str], actions: np.ndarray):
        """ Set actions for names
        actions is produced by an Agent instance, expected to be of shape [N_ENVS, N_UNITS]
        """

        def to_unity_action(name, action):
            # cont_a = np.zeros(4)
            cont_a = np.hstack((action[0:4].reshape((1, 4)), np.zeros((1, 2))))
            if name.startswith('E0_Red_0'):
                dis_a = np.zeros((1, 4), np.int32)
                dis_a[0][0] = 0 if action[4] < 0.5 else 1
            else:
                if name.startswith('E0_Red'):
                    dis_a = np.zeros((1, 2), np.int32)
            if name.startswith('E0_Blue'):
                dis_a = np.zeros((1, 4), np.int32)
                dis_a[0][0] = 0 if action[4] < 0.5 else 1

            return cont_a, dis_a

        for uid, n in enumerate(self._red_names[0]):
            if uid == 0:
                action = actions[0][0][0]
            else:
                action = actions[1][0][uid - 1]
            cont_action, dist_action = to_unity_action(n, action)
            action_tuple = self.env.get_action_tuple()
            action_tuple.add_discrete(dist_action)
            action_tuple.add_continuous(cont_action)
            self.env.set_actions(n, action_tuple)

    def _get_done_and_reward(self):
        """  获取游戏逻辑上done和reward
                """
        done = np.array([False] * self.n_envs)
        reward = np.zeros((self.n_envs, self.n_planes), np.float32)
        penalty = np.zeros((self.n_envs, self.n_planes), np.float32)

        for i in range(self.n_envs):
            # 蓝方主机都死亡、红方主机未死亡且没有蓝方的蛋正在飞行
            if self.alive_units[i][-1] != 1 and self.alive_units[i][-2] != 1 and self.alive_units[i][0] == 1:
                done[i] = True
                reward[i][0:5] += self.reward_config['blue_dead_reward']
                reward[i][5:] -= self.reward_config['blue_dead_reward']

                self._win_rate[i] += 1

            # 红方主机死亡
            elif self.alive_units[i][0] != 1:
                done[i] = True
                reward[i][0:5] += self.reward_config['main_dead_reward']
                reward[i][5:] -= self.reward_config['main_dead_reward']

                # self.alive_steps[i][0] = self._epslen[i]
                self._lose_rate[i] += 1
                self.alive_units[i][0] = -1

            if done[i] == 0:
                for j in range(len(self._blue_names[0])):
                    if self.alive_units[i][j + 5] == 0:
                        reward[i][0:5] += self.reward_config['blue_dead_reward']
                        reward[i][j+5] -= self.reward_config['blue_dead_reward']

                        self.alive_units[i][j + 5] = -1

                for j in range(len(self._red_names[0])):
                    if self.alive_units[i][j] == 0:
                        reward[i][j] += -0.1
                        self.alive_units[i][j] = -1
                    if self.locked_warning[i][j] == 0:
                        if abs(self.dis[j][0]) > 0.01:
                            reward[i][j] -= 0.05 * abs(self.dis[j][0])
                            penalty[i][j] -= 0.05 * abs(self.dis[j][0])
                        if abs(self.dis[j][1]) > 0.02:
                            reward[i][j] -= 0.005 * abs(self.dis[j][1])
                            penalty[i][j] -= 0.005 * abs(self.dis[j][1])
                        if abs(self.dis[j][2]) > 0.02:
                            reward[i][j] -= 0.05 * abs(self.dis[j][2])
                            penalty[i][j] -= 0.05 * abs(self.dis[j][2])
                for j in range(len(self._blue_names[0])):
                    if self.locked_warning[i][j+5] == 0:
                        if abs(self.dis[j+5][0]) > 0.01:
                            reward[i][j+5] -= 0.05 * abs(self.dis[j+5][0])
                            penalty[i][j+5] -= 0.05 * abs(self.dis[j+5][0])
                        if abs(self.dis[j+5][1]) > 0.02:
                            reward[i][j+5] -= 0.005 * abs(self.dis[j+5][1])
                            penalty[i][j+5] -= 0.005 * abs(self.dis[j+5][1])
                        if abs(self.dis[j+5][2]) > 0.02:
                            reward[i][j+5] -= 0.05 * abs(self.dis[j+5][2])
                            penalty[i][j+5] -= 0.05 * abs(self.dis[j+5][2])

        self.pre_reward = reward.copy()
        return done, reward, penalty# , detect_r


""" Test Code """
if __name__ == '__main__':
    config = dict(
        env_name='dummy',
        uid2aid=[0, 1, 1, 1, 1, 2, 2],
        max_episode_steps=1000,
        n_envs=1,
        unity_config={
            # 'worker_id': 0,
            # 'file_name':'E:\FlightCombat\FightSimulator\FightSimulator\Packages\Demo\T2.exe'
        },
        reward_config={
            'detect_reward': 0.1, 'main_dead_reward': -10, 'blue_dead_reward': 10, 'grid_reward': 0.1
        }
    )


    def print_dict(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f'{prefix} {k}')
                print_dict(v, prefix + '\t')
            elif isinstance(v, tuple):
                # namedtuple is assumed
                print(f'{prefix} {k}')
                print_dict(v._asdict(), prefix + '\t')
            else:
                print(f'{prefix} {k}: {v}')


    def print_dict_info(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f'{prefix} {k}')
                print_dict_info(v, prefix + '\t')
            elif isinstance(v, tuple):
                # namedtuple is assumed
                print(f'{prefix} {k}')
                print_dict_info(v._asdict(), prefix + '\t')
            else:
                print(f'{prefix} {k}: {v.shape} {v.dtype}')


    n_unity_file = 1
    n_unity_env = []
    for n in range(n_unity_file):
        n_unity_env.append(UnityEnv(**config))
        # config['unity_config']['worker_id'] = config['unity_config']['worker_id'] + 1

    # assert False
    env = n_unity_env[0]
    observations = env.reset()
    print('reset observations')
    # for i, o in enumerate(observations):
    #    print_dict_info(o, f'\tagent{i}')

    for k in range(1, 50000):
        # env.env.reset_envs_with_ids([2])
        actions = env.random_action()
        # print(f'Step {k}, random actions', actions)
        observations, rewards, dones, reset = env.step(actions)
        # print(f'Step {k}, observations')
        # for i, o in enumerate(observations):
        #    print_dict_info(o, f'\tagent{i}')
        # print(f'Step {k}, rewards', rewards)
        # print(f'Step {k}, dones', dones)
        # print(f'Step {k}, reset', reset)
        info = env.info()
        # print(f'Step {k}, info')
        # for aid, i in enumerate(info):
        #    print_dict(i, f'\tenv{aid}')
