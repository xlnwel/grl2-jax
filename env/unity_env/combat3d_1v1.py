import math
import os
import sys
import time
import platform

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cloudpickle
import functools
from typing import List
from core.elements.builder import ElementsBuilder
from env.typing import EnvOutput
import numpy as np
import gym
from env.unity_env.interface import UnityInterface
from run.utils import search_for_config
from core.tf_config import configure_gpu

"""  Naming Conventions:
An Agent is the one that makes decisions, it's typically a RL agent
A Unit is a controllable unit in the environment.
Each agent may control a number of units.
All ids start from 0 and increases sequentially
"""

EDGE_X = [-50000, 50000]
EDGE_Y = [2000, 20000]
EDGE_Z = [-80000, 80000]
MISSILE_PENALTY = 0.1
TURN_PENALTY = 0.03
RADAR_REWARD = 0.5
FLY_NAME = '0819'
UPPER_INTERVAL = 1
MAX_V = 0.544  # max 1.3 min 0.7
MIN_V = 0.238
THETA_RANGE = [-1, 1]
PHI_RANGE = [-0.08, 0.08]
ROLL_RANGE = [-0.5, 0.5]
LOCK_PENALTY = 0.1
DIS_REWARD = 0.01
LOCK_REWARD = 0.5

TOLERANCE = 1000

MAX_SPEED = 686
MIN_SPEED = 140
MAX_ANGLE_SPEED = 10
MIN_ANGLE_SPEED = 0
MAX_ANGLE = 180


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

def x2tri(x):
    return np.cos(x), np.sin(x)


def xyz2tri(x, y, z, return_length=False):
    r = np.linalg.norm([x, y, z])
    theta = np.arccos(z / r) if r != 0 else 0
    phi = np.arctan2(y, x)
    if return_length:
        return np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi), r
    else:
        return np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    
def get_velocity(state):
    vel = state[..., 5:8]
    vel_scalar = np.linalg.norm(vel, -1)
    assert np.all(vel_scalar < MAX_SPEED), vel
    assert np.all(vel_scalar > MIN_SPEED), vel
    return vel / 1000, vel_scalar

def get_angle_velocity(state):
    av = state[..., 8:11]
    assert np.all(av <= MAX_ANGLE_SPEED), av
    assert np.all(av >= -MAX_ANGLE_SPEED), av
    return av

def get_angle(state):
    angel = state[..., 11:14]
    assert np.all(angel <= MAX_ANGLE), angel
    assert np.all(angel >= -MAX_ANGLE), angel
    return angel

def get_height(state):
    h = state[..., 3:4]
    assert 1900 < h < 20100, h
    return h / 1000

def get_overload(state):
    o = state[..., 14:15]
    assert -3 < o < 9, o
    return o

def clip_action(action):
    return np.clip(action, -1, 1)

# NOTE: Keep the class name fixed; do not invent a new one!
# We do not rely on this for distinction!
class UnityEnv:
    def __init__(
            self,
            uid2aid,
            n_envs,
            fly_control_dir, 
            unity_config,
            seed=None,
            frame_skip=10,
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
        self.frame_skip = frame_skip
        self.n_red_main = 1
        self.n_blue_main = 1
        self.n_planes = 2

        self.fly_control_dir = fly_control_dir

        self._seed = np.random.randint(1000) if seed is None else seed
        self.n_envs = n_envs
        self.unity_config = unity_config
        self.max_episode_steps = kwargs['max_episode_steps']
        self.reward_config = reward_config
        self.use_action_mask = False  # if action mask is used
        self.use_life_mask = False  # if life mask is used
        if platform.system() == 'Windows':
            self.unity_config['file_name'] = None
            self.unity_config['worker_id'] = 0

        self.action_dim = [4, 4]
        self.action_space = [gym.spaces.Box(low=-1, high=1, shape=(ad,)) for ad in self.action_dim]
        self._obs_dim = [30, 30]
        self._global_state_dim = [30, 30]

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

        self._dense_score = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._turning_penalty = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._out_penalty = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._locked_penalty = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._dis_reward = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._lock_reward = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        # The length of the episode
        self._epslen = np.zeros(self.n_envs, np.int32)

        # 红蓝双方存活状况
        self.alive_units = np.ones((self.n_envs, self.n_planes), np.int32)

        self.pre_reward = np.zeros((self.n_envs, self.n_units), np.float32)
        self.prev_action = None
        self.fly_red = None
        # 策略输出的动作执行1次
        self.fly_control_steps = UPPER_INTERVAL
        self.shot_steps = 0
        self.unity_shot_steps = 0
        self.total_step_time = 0
        self.env_step_time = 0
        self.locked_warning = np.zeros((self.n_envs, self.n_planes), bool)
        self.shot_warning = np.zeros((self.n_envs, self.n_planes), bool)
        self.radar_locked = np.zeros((self.n_envs, self.n_planes), bool)
        # 蛋的状态，-1未发射，0-6是飞行中，代表其目标，-2表示已爆炸
        self.missile_state = np.zeros((self.n_envs, 3, 4))
        # 蛋的位置信息
        self.missile_position = np.zeros((self.n_envs, 3, 4))

    def random_action(self):
        actions = []
        for aid, uids in enumerate(self.aid2uids):
            a = np.array([[self.action_space[aid].sample() for _ in uids] for _ in range(self.n_envs)], np.float32)
            actions.append(a)
        return actions

    def _step(self):
        _, decision_steps, _ = self.env.step()
        names = self.env.get_behavior_names()
        assert len(names) == 1, names
        self.decision_steps = decision_steps[names[0]]
        return self.decision_steps

    def reset(self):
        self.env = UnityInterface(**self.unity_config)
        self.blue_radar_action = -1 * np.ones((1, 1))
        self.blue_shot_action = np.zeros((1, 1))
        self._win_rate = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._lose_rate = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._draw_rate = np.zeros((self.n_envs, self.n_units), dtype=np.float32)

        self._dense_score = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._turning_penalty = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._out_penalty = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._locked_penalty = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._dis_reward = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._lock_reward = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)

        self._detect_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self._missile_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)

        self._epslen = np.zeros(self.n_envs, np.int32)

        self.shot_steps = 0

        # 设置发蛋间隔在20步（20s）
        self.red_shot_time = 0
        self.blue_shot_time = 0

        # alive记录，1位存活，0为刚被击毁，-1为已被击毁
        self.alive_units = np.ones((self.n_envs, self.n_planes), np.int32)
        self.detect_units = np.zeros((self.n_envs, self.n_units), bool)

        self.pre_reward = np.zeros((self.n_envs, self.n_units), np.float32)
        self.prev_action = None
        self.fly_red = self.init_red_fly_model()
        self.dis = np.zeros((2, 4))
        self.last_position = np.zeros((self.n_envs, self.n_units, 3))
        self.position = np.zeros((self.n_envs, self.n_units, 3))

        self.env.reset()
        self._step()
        
        return self._get_obs()

    def init_red_fly_model(self):
        ubuntu_dir = f"/home/ubuntu/chenxinwei/grl/unity-logs/unity-fly_control/sync2-zero/{self.fly_control_dir}/seed=None/a0/i1-v1"
        if platform.system() == 'Windows':
            directory = [f"D:\FlightCombat\CombatTrain/logs/unity-fly_control/sync-hm/{self.fly_control_dir}/seed=None/a0/i1-v1"]
        else:
            directory = [ubuntu_dir]

        configs = [search_for_config(d) for d in directory]
        config = configs[0]
        env_stats = {'obs_shape': [{'obs': (15,), 'global_state': (15,), 'prev_reward': (), 'prev_action': (5,)}],
                     'obs_dtype': [{'obs': np.float32, 'global_state': np.float32, 'prev_reward': np.float32,
                                    'prev_action': np.float32}],
                     'action_shape': [(5,)], 'action_dim': [5], 'action_low': None, 'action_high': None,
                     'is_action_discrete': [False], 'action_dtype': [np.float32], 'n_agents': 1, 'n_units': 1,
                     'uid2aid': [0], 'aid2uids': [np.array([0])], 'use_life_mask': False, 'use_action_mask': False,
                     'is_multi_agent': True, 'n_workers': 1, 'n_envs': 1}
        self.fly_obs_shape = [dict(
            obs=(15,),
            global_state=(15,),
            prev_reward=(),
            prev_action=(5,),
        )]
        self.fly_obs_dtype = [dict(
            obs=np.float32,
            global_state=np.float32,
            prev_reward=np.float32,
            prev_action=np.float32,
        )]

        builder = ElementsBuilder(config, env_stats, to_save_code=False)
        elements = builder.build_acting_agent_from_scratch(to_build_for_eval=True)
        agent = elements.agent

        return agent

    def get_fly_obs(self, ds, target_v):
        _, v_scalar = get_velocity(ds)
        angle_v = get_angle_velocity(ds)
        angle = get_angle(ds)

        theta = angle[1]
        phi = angle[0]
        roll = angle[2]

        now_v = np.stack([v_scalar, theta, phi, roll], -1)
        dis = target_v - now_v
        dis[..., 1:] = dis[..., 1:] - np.copysign(dis[..., 1:], 2)
        assert dis.shape == (self.n_envs, self.n_units, 4), dis.shape

        v_ = np.expand_dims(v_scalar, -1)
        posture = np.concatenate(
            x2tri(np.deg2rad(theta)) + x2tri(np.deg2rad(phi)) + x2tri(np.deg2rad(roll))
        , -1)
        assert posture.shape == (self.n_envs, self.n_units, 6), dis.shape
        height = get_height(ds)
        assert height.shape == (self.n_envs, self.n_units, 1), height.shape

        obs = np.concatenate((
            v_,         # 1
            angle_v,    # 3
            posture,    # 6 
            height,     # 1
            dis,        # 4
        ), -1)
        observations = {}
        observations['obs'] = obs
        observations['global_state'] = obs
        all_obs = [observations]

        return all_obs, now_v

    def get_fly_control_goal(self, action):
        goal = np.zeros(action.shape[0], 4)
        goal[0] = (action[:, 0] + 1) * (MAX_V - MIN_V) / 2 + MIN_V
        goal[1] = action[:, 1]
        goal[2] = action[:, 2] * PHI_RANGE[1]
        return goal

    def send_action(self, actions):
        for _ in range(self.fly_control_steps):
            target_v = np.concatenate([a[:3] for a in actions], 1)
            target_v = np.pad(target_v, [(0, 0), (0, 0), (0, 1)])
            assert target_v.shape == (self.n_envs, self.n_units, 4), target_v.shape
            assert np.all(target_v[..., -1] == 0), target_v[..., -1]

            names = self.env.get_behavior_names()
            assert len(names) == 1, names
            prev_obs = np.array([o[0] for o in self.decision_steps.obs])
            prev_obs = prev_obs.reshape(self.n_envs, self.n_units, 133)
            assert prev_obs.shape == (self.n_envs, self.n_units, 133), prev_obs.shape

            fc_obs, now_v = self.get_fly_obs(prev_obs, target_v)
            self.dis = target_v - now_v
            env_output = EnvOutput(fc_obs, None, None, None)
            fc_act = self.fly_red(env_output, evaluation=True)

            for i in range(self.frame_skip):
                # TODO: fill in other actions
                if i == 0:
                    pass
                else:
                    pass
                self.env.set_actions(fc_act)

                start1 = time.time()
                self._step()
                end1 = time.time()
                self.env_step_time += end1 - start1

    def step(self, actions):
        self.send_action(actions)

        self._epslen += 1

        agent_obs = self._get_obs(self.decision_steps, self.pre_reward)

        done, reward, out_penalty, turning_penalty, locked_penalty, dis_reward, lock_reward = self._get_done_and_reward()

        for i in range(self.n_envs):
            if self._epslen[i] > self.max_episode_steps:
                done[i] = True
                self._draw_rate[i] += 1

        discount = 1 - done  # we return discount instead of done

        assert reward.shape == (self.n_envs, self.n_planes), reward.shape
        assert discount.shape == (self.n_envs,), discount.shape

        rewards = reward
        discounts = np.tile(discount.reshape(-1, 1), (1, self.n_planes))

        self._dense_score += rewards
        self._turning_penalty += turning_penalty
        self._out_penalty += out_penalty
        self._locked_penalty += locked_penalty
        self._dis_reward += dis_reward
        self._lock_reward += lock_reward

        reset = done
        assert reset.shape == (self.n_envs,), reset.shape
        resets = np.tile(reset.reshape(-1, 1), (1, self.n_units))

        self.prev_action = actions

        alive_blue = np.zeros(self.n_envs)
        alive_main = np.zeros(self.n_envs)
        alive_blue[0] = 1 - (self.alive_units[0][-1] != 1)
        alive_main[0] = 1 - (self.alive_units[0][0] != 1)

        self._info = [dict(
            score=self._win_rate[i].copy(),
            win_rate=self._win_rate[i].copy(),
            lose_rate=self._lose_rate[i].copy(),
            draw_rate=self._draw_rate[i].copy(),
            dense_score=self._dense_score[i].copy(),
            turning_penalty=self._turning_penalty[i].copy(),
            out_penalty=self._out_penalty[i].copy(),
            locked_penalty=self._locked_penalty[i].copy(),
            lock_reward=self._lock_reward[i].copy(),
            dis_reward=self._dis_reward[i].copy(),
            epslen=np.array([self._epslen[i]] * self.n_units),
            game_over=np.array([discount[i] == 0] * self.n_units),
            left_missile_red=np.hstack([self.red_missile_left[i][0][0]] * self.n_units),
            left_missile_blue0=np.array([self.blue_missile_left[i][0][0]] * self.n_units),
            alive_blue=np.array([alive_blue[i]] * self.n_units),
            alive_main=np.array([alive_main[i]] * self.n_units),
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
                self._turning_penalty[i] = np.zeros(self.n_units, np.float32)
                self._out_penalty[i] = np.zeros(self.n_units, np.float32)
                self._locked_penalty[i] = np.zeros(self.n_units, np.float32)
                self._dis_reward[i] = np.zeros(self.n_units, np.float32)
                self._lock_reward[i] = np.zeros(self.n_units, np.float32)
                self._win_rate[i] = np.zeros(self.n_units, np.float32)
                self._lose_rate[i] = np.zeros(self.n_units, np.float32)
                self._draw_rate[i] = np.zeros(self.n_units, np.float32)
                self.alive_units[i] = np.ones(self.n_planes, np.int32)
                self.locked_warning = np.zeros((self.n_envs, 2), bool)
                self.shot_warning = np.zeros((self.n_envs, self.n_planes), bool)
                self.radar_locked = np.zeros((self.n_envs, self.n_planes), bool)
                self._epslen[i] = 0
                self.red_shot_time = 0
                self.blue_shot_time = 0
                self.prev_action = None
        if len(reset_env) != 0:
            self.env.reset()
            reset, decision_steps, terminal_steps = self.env.step()
            self.decision_steps = decision_steps

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

    def _get_obs(self, reward=None, action=None):
        all_obs = self._process_obs(reward, action)

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

    def _construct_obs(self, plane_infos, team, side, i):
        plane = plane_infos[i, 0:21]
        weapon = plane_infos[i, 21:25]
        radar = plane_infos[i, 109:]
        # radar = plane_infos[i, 25:33]

        obs = []

        alive = plane[1]
        position = plane[2:5]
        angle = plane[5:8]
        obs.extend(x2tri(angle[0]) + x2tri(angle[1]) + x2tri(angle[2]))
        dis_x = np.min([position[0] - EDGE_X[0], EDGE_X[1] - position[0]])
        dis_y = np.min([position[1] - EDGE_Y[0], EDGE_Y[1] - position[1]])
        dis_z = np.min([position[2] - EDGE_Z[0], EDGE_Z[1] - position[2]])
        edge_dis = [dis_x, dis_y, dis_z]
        obs.extend(edge_dis)
        self.alive_units[team][i] = alive if self.alive_units[team][i] != -1 else -1
        vel = plane[5:8]
        obs.extend(vel)
        obs.append(np.linalg.norm(vel))

        #angle = red_self[i][11:14]
        # for tt in range(3):
        #     while abs(angle[tt]) > 180:
        #         angle[tt] = angle[tt] - math.copysign(360, angle[tt])
        overload = plane[17]
        obs.append(overload)
        #overload = np.clip(overload, -2, 2)

        # oil = plane[18]
        locked_warning = plane[19]
        shot_warning = plane[20]
        obs.extend([locked_warning, shot_warning])
        self.locked_warning[team][i] = locked_warning
        self.shot_warning[team][i] = shot_warning

        # me_missile = self._get_target_missile(blue_weapon, team, i, positions[i])
        # TODO: This should be modified when there are more than two planes
        enemy_dist = self.position[1-side] - self.position[side]
        obs.extend(enemy_dist)
        # threshold = 1000
        # oil_warning = 1 if oil < threshold else 0
        # which_missile = weapon[i][0] if i < self.n_red_main else 0
        mid_dis_missile = weapon[1] if i < self.n_red_main else 0
        # short_dis_missile = weapon[i][2] if i < self.n_red_main else 0
        # disrupt_missile = weapon[i][3] if i < self.n_red_main else 0
        # alive_blue_num = sum(blue_self[:, 1])
        obs.extend(
            self.to_one_hot(int(mid_dis_missile), 5)
        )

        #自己蛋的信息 冗余
        # if i < self.n_red_main:
        #     self_miss_info = []
        #     self_miss_state = []
        #     for index in range(4, len(red_main_weapon[i]), 14):
        #         miss_type = np.array([red_main_weapon[i][index]])
        #         miss_pos = red_main_weapon[i][index + 1:index + 4]
        #         miss_vel = red_main_weapon[i][index + 4:index + 7]
        #         miss_angle = red_main_weapon[i][index + 7:index + 10]
        #
        #         miss_direction = red_main_weapon[i][index + 10:index + 13]
        #         miss_state = np.array([red_main_weapon[i][index + 13]])
        #         self_miss_info.append(miss_type)
        #         self_miss_info.append(miss_pos)
        #         self_miss_info.append(miss_vel)
        #         self_miss_info.append(
        #             np.concatenate([np.sin(np.deg2rad(miss_angle)), np.sin(np.deg2rad(miss_angle))]))
        #         self_miss_state.append(miss_state)
        #     missile_cont = np.hstack(self_miss_info).flatten()
        #     missile_disc = np.hstack(self_miss_state).flatten()
        #     obs.append(missile_cont)
        #     obs.append(missile_disc)
        # else:
        #    obs.append(np.zeros(84))
        # radar_state = self.to_one_hot(int(red_radar[i][0]), 3)
        radar_target = self.to_one_hot(int(radar[1]) + 1, 3)
        # self.radar_locked[team][i] = 1 if radar[1] != -1 else 0
        self.radar_locked[team][i] = radar[4]
        obs.extend(radar_target)

        # radar_detect = radar[4]
        radar_detect_yxz = radar[5:8]
        obs.extend(radar_detect_yxz)
        # print(obs)

        # all_action_mask.append(np.ones(self.action_dim[self.uid2aid[i]], bool))

        obs = np.array(obs, np.float32) # 30
        if not alive:
            obs = np.zeros_like(obs)
        return obs

    def _process_obs(self, reward, action):
        obs = np.array([o[0] for o in self.decision_steps.obs])
        ids = np.argsort(self.decision_steps.agent_id)
        obs = obs[ids].reshape(self.n_envs, self.n_units, 133)
        assert obs.shape == (self.n_envs, self.n_units, 133), obs.shape

        red_obs = obs[:, 0]
        blue_obs = obs[:, 1]

        red_self = red_obs[..., 0:21]
        red_main_weapon = red_obs[..., 21:109]

        blue_self = blue_obs[:, 0:21]
        blue_weapon = blue_obs[:, 21:109]

        all_env_obs = []
        self.blue_missile_left = np.zeros((self.n_envs, self.n_blue_main, 3), np.int32)
        self.red_missile_left = np.zeros((self.n_envs, self.n_red_main, 3), np.int32)

        self.detect_units = np.zeros((self.n_envs, self.n_units), bool)

        # 根据状态信息获取获取各个飞机相互之间的相对坐标、距离、角色等
        all_self = np.concatenate((red_self, blue_self), axis=0)

        positions = self._get_all_positions(all_self)
        self.last_position = self.position.copy()
        self.position = positions
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
            obs = self._construct_obs(plain_info, team, 0, i)
            global_state = obs

            all_obs.append(obs.copy())
            all_global_state.append(global_state.copy())
            all_alive.append(self.alive_units[team][i])
            prev_action = np.zeros((self.action_dim[self.uid2aid[i]]), np.float32)
            all_prev_action.append(prev_action)

        for i, name in enumerate(self._blue_names[team]):
            i = sum([len(self.aid2uids[k]) for k in range(1)]) + i
            obs = self._construct_obs(plain_info, team, 1, i)
            global_state = obs

            all_obs.append(obs.copy())
            all_global_state.append(global_state.copy())
            all_alive.append(self.alive_units[team][i])
            prev_action = np.zeros((self.action_dim[self.uid2aid[i]]), np.float32)
            all_prev_action.append(prev_action)

        observations['obs'] = all_obs
        observations['global_state'] = all_global_state
        observations['life_mask'] = all_alive
        observations['action_mask'] = all_action_mask
        observations['prev_reward'] = np.zeros(self.n_units, np.float32) if reward is None else reward[team]
        observations['prev_action'] = all_prev_action

        all_env_obs.append(observations)

        return all_env_obs

    def to_one_hot(self, m, n):
        one_hot = [0] * n
        one_hot[m] = 1
        return one_hot

    def _get_all_positions(self, all_states):
        pos = np.array(all_states[..., 2:5])
        assert np.all(pos[0] <= EDGE_X[1] + TOLERANCE)
        assert np.all(pos[0] >= EDGE_X[0] - TOLERANCE)
        assert np.all(pos[1] <= EDGE_Y[1] + TOLERANCE)
        assert np.all(pos[1] >= EDGE_Y[0] - TOLERANCE)
        assert np.all(pos[2] <= EDGE_Z[1] + TOLERANCE)
        assert np.all(pos[2] >= EDGE_Z[0] - TOLERANCE)
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
            cont_a = np.hstack((move_action[0:5].reshape((1, 5)), np.zeros((1, 2))))
            if name.startswith('E0_Red_0'):
                dis_a = np.zeros((1, 4), np.int32)
                if shot_action < 0.5:
                    dis_a[0][0] = 0
                else:
                    dis_a[0][0] = 1
                    self.unity_shot_steps += 1
                dis_a[0][2] = radar_action
            if name.startswith('E0_Blue'):
                dis_a = np.zeros((1, 4), np.int32)
                dis_a[0][0] = 0 if shot_action < 0.5 else 1
                dis_a[0][2] = radar_action

            return cont_a, dis_a

        for uid, n in enumerate(names[0]):
            move_action = move_actions[n][0][0][0]
            move_action = np.clip(move_action, -1, 1)
            move_action[4] = (move_action[4] + 1)/2

            radar_action = radar_actions[n]
            shot_action = shot_actions[n]
            cont_action, dist_action = to_unity_action(n, move_action, radar_action, shot_action)
            action_tuple = self.env.get_action_tuple()
            action_tuple.add_discrete(dist_action)
            action_tuple.add_continuous(cont_action)
            self.env.set_actions(n, action_tuple)

    def _get_done_and_reward(self):
        """  获取游戏逻辑上done和reward
                """
        done = np.array([False] * self.n_envs, np.float32)
        reward = np.zeros((self.n_envs, self.n_planes), np.float32)
        out_penalty = np.zeros((self.n_envs, self.n_planes), np.float32)
        turning_penalty = np.zeros((self.n_envs, self.n_planes), np.float32)
        locked_penalty = np.zeros((self.n_envs, self.n_planes), np.float32)
        dis_reward = np.zeros((self.n_envs, self.n_planes), np.float32)
        lock_reward = np.zeros((self.n_envs, self.n_planes), np.float32)

        for i in range(self.n_envs):
            if self.alive_units[i][-1] != 1 and self.alive_units[i][0] == 1:
                done[i] = True
                reward[i][0] += self.reward_config['blue_dead_reward']
                reward[i][1] -= self.reward_config['blue_dead_reward']
                self._win_rate[i] += 1

            # 红方主机死亡
            if self.alive_units[i][0] != 1 and self.alive_units[i][-1] == 1:
                done[i] = True
                reward[i][0] += self.reward_config['main_dead_reward']
                reward[i][1] -= self.reward_config['main_dead_reward']
                # self.alive_steps[i][0] = self._epslen[i]
                self._lose_rate[i] += 1
                self.alive_units[i][0] = -1

            if done[i] == 0:
                for id, p in enumerate(self.position):
                    if p[0] > EDGE_X[1] or p[0] < EDGE_X[0] or p[1] > EDGE_Y[1] or p[1] < EDGE_Y[0] or p[2] > EDGE_Z[1] or p[2] < EDGE_Z[0]:
                        done[i] = True
                        reward[i][id] += self.reward_config['main_dead_reward'] - 2
                        out_penalty[i][id] += self.reward_config['main_dead_reward'] - 2

                for j in range(len(self._red_names[0])):
                    if self.radar_locked[i][j] == 1:
                        reward[i][j] += LOCK_REWARD
                        lock_reward[i][j] += LOCK_REWARD

                    if self.locked_warning[i][j] == 0:
                        if abs(self.dis[j][0]) > 0.01 or abs(self.dis[j][1]) > 0.02 or abs(self.dis[j][2]) > 0.02:
                            reward[i][j] -= TURN_PENALTY
                            turning_penalty[i][j] -= TURN_PENALTY

                        #     reward[i][j] -= TURN_PENALTY * abs(self.dis[j][0])
                        #     turning_penalty[i][j] -= TURN_PENALTY * abs(self.dis[j][0])
                        # if abs(self.dis[j][1]) > 0.02:
                        #     reward[i][j] -= 0.1 * TURN_PENALTY * abs(self.dis[j][1])
                        #     turning_penalty[i][j] -= 0.1 * TURN_PENALTY * abs(self.dis[j][1])
                        # if abs(self.dis[j][2]) > 0.02:
                        #     reward[i][j] -= TURN_PENALTY * abs(self.dis[j][2])
                        #     turning_penalty[i][j] -= TURN_PENALTY * abs(self.dis[j][2])
                    else:
                        reward[i][j] -= LOCK_PENALTY
                        locked_penalty[i][j] -= LOCK_PENALTY

                for j in range(len(self._blue_names[0])):
                    if self.radar_locked[i][j + 1] == 1:
                        reward[i][j + 1] += LOCK_REWARD
                        lock_reward[i][j + 1] += LOCK_REWARD

                    if self.locked_warning[i][j + 1] == 0:
                        if abs(self.dis[j+1][0]) > 0.01 or abs(self.dis[j+1][1]) > 0.02 or abs(self.dis[j+1][2]) > 0.02:
                            reward[i][j+1] -= TURN_PENALTY
                            turning_penalty[i][j+1] -= TURN_PENALTY
                        # if abs(self.dis[j + 1][0]) > 0.01:
                        #     reward[i][j + 1] -= TURN_PENALTY * abs(self.dis[j + 1][0])
                        #     turning_penalty[i][j + 1] -= TURN_PENALTY * abs(self.dis[j + 1][0])
                        # if abs(self.dis[j + 1][1]) > 0.02:
                        #     reward[i][j + 1] -= 0.1 * TURN_PENALTY * abs(self.dis[j + 1][1])
                        #     turning_penalty[i][j + 1] -= 0.1 * TURN_PENALTY * abs(self.dis[j + 1][1])
                        # if abs(self.dis[j + 1][2]) > 0.02:
                        #     reward[i][j + 1] -= TURN_PENALTY * abs(self.dis[j + 1][2])
                        #     turning_penalty[i][j + 1] -= TURN_PENALTY * abs(self.dis[j + 1][2])
                    else:
                        reward[i][j + 1] -= LOCK_PENALTY
                        locked_penalty[i][j+1] -= LOCK_PENALTY

                last_dis = np.linalg.norm(self.last_position[0] - self.last_position[1])
                now_dis = np.linalg.norm(self.position[0] - self.position[1])
                if now_dis < last_dis:
                    reward[i] += DIS_REWARD
                    dis_reward += DIS_REWARD

        # if self._epslen[0] != 1 and self.blue_missile_left[0][0][0] == 0 and self.red_missile_left[0][0][0] == 0:
        #     done[i] = True

        self.pre_reward = reward.copy()
        return done, reward, out_penalty, turning_penalty, locked_penalty, dis_reward, lock_reward  # , detect_r


""" Test Code """
if __name__ == '__main__':
    config = dict(
        env_name='dummy',
        uid2aid=[0, 1],
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


    from tools import yaml_op
    config = yaml_op.load_config('distributed/sync2/configs/unity3_1v1.yaml')
    config = config['env']
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
    all_start = time.time()
    for k in range(1, 100):
        # env.env.reset_envs_with_ids([2])
        #actions = env.random_action()
        actions = [np.array(([[[0.4, 0, 0.05, 1]]])), np.array(([[[0.4, 0, 0.05, 1]]]))]
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
    # all_end = time.time()
    # print('all 100 steps = ' + str(all_end - all_start))
    # print('env step 100 steps = ' + str(env.env_step_time))
    # print('unity step 100 steps = ' + str(env.env.env_step_time))