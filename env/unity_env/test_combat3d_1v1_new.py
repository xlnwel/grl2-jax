import math
import os
import sys
import time
import platform

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
RADAR_REWARD = 0
FLY_NAME = '0902-fs=10_1copy-lr=1e-3'
UPPER_INTERVAL = 1
MAX_V = 0.544  # max 1.3 min 0.7
MIN_V = 0.238
THETA_RANGE = [-1, 1]
PHI_RANGE = [-0.08, 0.08]
ROLL_RANGE = [-0.5, 0.5]
LOCK_PENALTY = 0.03
DIS_REWARD = 0.01
LOCK_REWARD = 0.05

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
    return np.stack((np.cos(x), np.sin(x)), axis=-1)


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
    vel_scalar = np.linalg.norm(vel, axis=-1)
    # assert np.all(vel_scalar < MAX_SPEED), vel
    # assert np.all(vel_scalar > MIN_SPEED), vel
    return vel / 1000, vel_scalar / 1000


def get_angle_velocity(state):
    av = state[..., 8:11]
    # assert np.all(av <= MAX_ANGLE_SPEED), av
    # assert np.all(av >= -MAX_ANGLE_SPEED), av
    return av


def get_angle(state):
    angel = state[..., 11:14]
    # assert np.all(angel <= MAX_ANGLE), angel
    # assert np.all(angel >= -MAX_ANGLE), angel
    return angel


def get_height(state):
    h = state[..., 3:4]
    # assert np.all(h < 21000), h
    # assert np.all(h > 1500), h

    return h / 1000


def get_xyz(state):
    xyz = state[..., 2:5]
    return xyz / 1000


def get_overload(state):
    o = state[..., 14:15]
    # assert -3 < o < 9, o
    return o


def get_angle_diff(a1, a2):
    diff = abs(a1 - a2)
    return np.where(diff >= 1, 2-diff, diff)


def clip_action(action):
    return np.clip(action, -1, 1)


# NOTE: Keep the class name fixed; do not invent a new one!
# We do not rely on this for distinction!
class UnityEnv:
    def __init__(
            self,
            uid2aid,
            n_envs,
            unity_config,
            fly_control_dir=None,
            seed=None,
            frame_skip=10,
            is_action_discrete=True,
            reward_config={},
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

        self.fly_control_dir = FLY_NAME

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

        self._win_rate = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._lose_rate = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._draw_rate = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)

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
        self.shot_steps = np.zeros((self.n_envs, self.n_planes), dtype=np.int32)
        self.unity_shot_steps = np.zeros((self.n_envs, self.n_planes), dtype=np.int32)
        self.total_step_time = 0
        self.env_step_time = 0
        self.locked_warning = np.zeros((self.n_envs, self.n_planes), bool)
        self.shot_warning = np.zeros((self.n_envs, self.n_planes), bool)
        self.radar_locked = np.zeros((self.n_envs, self.n_planes), bool)
        self.name = 'Player?team=0'
        self.env = UnityInterface(**self.unity_config)
        self.dis = None
        self.blue_shot_time = None
        self.red_shot_time = None
        self.last_position = None
        self.position = None

    def random_action(self):
        actions = []
        for aid, uids in enumerate(self.aid2uids):
            a = np.array([[self.action_space[aid].sample() for _ in uids] for _ in range(self.n_envs)], np.float32)
            actions.append(a)
        return actions

    def fly_draw_line(self):
        actions = []
        for aid, uids in enumerate(self.aid2uids):
            a = np.array([[[0, 2, 0, 1, 2] for _ in uids] for _ in range(self.n_envs)], np.float32)
            actions.append(a)
        # if abs(self._v[0][0][3]) < 0.4:
        #     for aid, uids in enumerate(self.aid2uids):
        #         a = np.array([[[0, 0, 2, 1, 2] for _ in uids] for _ in range(self.n_envs)], np.float32)
        #         actions.append(a)
        # else:
        #     for aid, uids in enumerate(self.aid2uids):
        #         a = np.array([[[-2, 0, 0, 1, 2] for _ in uids] for _ in range(self.n_envs)], np.float32)
        #         actions.append(a)
        return actions

    def _step(self):
        _, decision_steps, _ = self.env.step()
        #names = self.env.get_behavior_names()
        #assert len(names) == 1, names
        self.decision_steps = decision_steps[self.name]
        return self.decision_steps

    def reset(self):
        self._win_rate = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._lose_rate = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._draw_rate = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)

        self._dense_score = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._turning_penalty = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._out_penalty = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._locked_penalty = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._dis_reward = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._lock_reward = np.zeros((self.n_envs, self.n_planes), dtype=np.float32)
        self._epslen = np.zeros(self.n_envs, np.int32)
        self.shot_steps = np.zeros((self.n_envs, self.n_planes), dtype=np.int32)

        # 设置发蛋间隔在20步（20s）
        self.red_shot_time = np.zeros((self.n_envs, self.n_red_main), dtype=np.int32)
        self.blue_shot_time = np.zeros((self.n_envs, self.n_blue_main), dtype=np.int32)

        # alive记录，1位存活，0为刚被击毁，-1为已被击毁
        self.alive_units = np.ones((self.n_envs, self.n_planes), np.int32)

        self.pre_reward = np.zeros((self.n_envs, self.n_units), np.float32)
        self.prev_action = None
        self.fly_red = self.init_red_fly_model()
        self.dis = np.zeros((self.n_envs, self.n_planes, 4))
        self.last_position = np.zeros((self.n_envs, self.n_units, 3))
        self.position = np.zeros((self.n_envs, self.n_units, 3))

        # self.env.reset()
        self._step()

        return self._get_obs()

    def init_red_fly_model(self):
        # ubuntu_dir = f"/home/ubuntu/chenxinwei/grl/unity-logs/unity-fly_control/sync2-zero/{self.fly_control_dir}/seed=None/a0/i1-v1"
        ubuntu_dir = f"/home/ubuntu/wuyunkun/hm/logs/unity-fly_control_old/sync-hm/{self.fly_control_dir}/seed=None/a0/i1-v1"
        if platform.system() == 'Windows':
            directory = [
                f"D:\FlightCombat\CombatTrain/logs/unity-combat_fly_control/sync-hm/{self.fly_control_dir}/seed=None/a0/i1-v1"]
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

        theta = angle[..., 1]
        phi = angle[..., 0]
        roll = angle[..., 2]

        now_v = np.stack([v_scalar, theta / 180, phi / 180, roll / 180], -1)
        dis = target_v - now_v
        dis[..., 1:] = dis[..., 1:] - np.copysign(dis[..., 1:], 2)
        # assert dis.shape == (self.n_envs, self.n_units, 4), dis.shape

        v_ = np.array([v_scalar, phi, roll])
        posture = np.concatenate([x2tri(np.deg2rad(phi)), x2tri(np.deg2rad(roll))], -1)
        v_no_theta = np.stack([v_scalar, phi / 180, roll / 180], -1)

        # posture = np.concatenate([
        #     x2tri(np.deg2rad(theta)), x2tri(np.deg2rad(phi)), x2tri(np.deg2rad(roll))
        # ], -1)
        # assert posture.shape == (self.n_envs, self.n_units, 6), dis.shape
        height = get_height(ds)
        assert height.shape == (self.n_envs, self.n_units, 1), height.shape
        obs = np.concatenate((
            v_no_theta,  # 3
            angle_v,  # 3
            posture,  # 4
            height / 20,  # 1
            dis,  # 4
        ), -1)
        obs = np.array(obs, dtype=np.float32)
        observations = {}
        observations['obs'] = obs
        observations['global_state'] = obs
        observations['life_mask'] = [1]
        observations['action_mask'] = [1, 1, 1, 1, 1]
        observations['prev_reward'] = np.zeros((self.n_envs, self.n_units, 1), np.float32)
        observations['prev_action'] = np.zeros((self.n_envs, self.n_units, self.action_dim[self.uid2aid[0]]), np.float32)

        r_obs = [{}]

        for aid in range(1):
            for k in self.fly_obs_shape[aid].keys():
                r_obs[aid][k] = np.zeros(
                    (self.n_envs, len(self.aid2uids[aid]), *self.fly_obs_shape[aid][k]),
                    dtype=self.fly_obs_dtype[aid][k]
                )
                r_obs[aid][k] = observations[k]

        return r_obs[0], now_v

    def get_fly_control_goal(self, action):
        goal = np.zeros(action.shape[0], 4)
        goal[0] = (action[:, 0] + 1) * (MAX_V - MIN_V) / 2 + MIN_V
        goal[1] = action[:, 1]
        goal[2] = action[:, 2] * PHI_RANGE[1]
        return goal

    def send_action(self, actions):
        # print(actions)
        for _ in range(self.fly_control_steps):
            target_v = np.concatenate([a[..., :3] for a in actions], 1)
            target_v = np.pad(target_v, [(0, 0), (0, 0), (0, 1)])
            assert target_v.shape == (self.n_envs, self.n_units, 4), target_v.shape
            assert np.all(target_v[..., -1] == 0), target_v[..., -1]

            names = self.env.get_behavior_names()
            assert len(names) == 1, names
            prev_obs = self.decision_steps.obs[0]
            ids = np.argsort(self.decision_steps.agent_id)
            prev_obs = prev_obs[ids].reshape(self.n_envs, self.n_units, 133)
            assert prev_obs.shape == (self.n_envs, self.n_units, 133), prev_obs.shape

            fc_obs, now_v = self.get_fly_obs(prev_obs, target_v)
            self.dis = target_v - now_v
            env_output = EnvOutput(fc_obs, None, None, None)
            fc_act = self.fly_red(env_output, evaluation=True)[0]

            for i in range(self.frame_skip):
                # TODO: fill in other actions
                if i == 0:
                    radar_action = np.ones((self.n_envs, self.n_units, 1))
                    red_shot_action = actions[0][..., -1]
                    blue_shot_action = actions[1][..., -1]
                    shot_action = np.zeros((self.n_envs, self.n_units, 1))

                    for team in range(self.n_envs):
                        for id in range(self.n_red_main):
                            if red_shot_action[team][id] > 0 and \
                                    self._epslen[team] - self.red_shot_time[team][id] > 20 and \
                                    self.red_missile_left[team][id][0] > 0:
                                #red_shot_action[team][id] = 1
                                xyz_dir = self.ds_obs[team][1][2:5] - self.ds_obs[team][0][2:5]
                                xyz_distance = np.linalg.norm(xyz_dir)
                                # if xyz_distance < 40000:
                                xyz_dir = xyz_dir / xyz_distance
                                    #print(xyz_dir)
                                direction = self.ds_obs[team][0][14:17]
                                    #print(direction)
                                    # print(np.dot(xyz_dir, direction))
                                shot_action[team][id] = 1
                                self.red_shot_time[team][id] = self._epslen[team]

                            else:
                                shot_action[team][id] = 0
                        for id in range(self.n_blue_main):
                            if blue_shot_action[team][id] > 0 and \
                                    self._epslen[team] - self.blue_shot_time[team][id] > 20 and \
                                    self.blue_missile_left[team][id][0] > 0:
                                shot_action[team][self.n_red_main + id] = 1
                                self.blue_shot_time[team][id] = self._epslen[team]
                            else:
                                shot_action[team][self.n_red_main + id] = 0
                else:
                    radar_action = np.zeros((self.n_envs, self.n_units, 1))
                    shot_action = np.zeros((self.n_envs, self.n_units, 1))

                self.set_actions(fc_act, radar_action, shot_action)

                start1 = time.time()
                self._step()
                end1 = time.time()
                self.env_step_time += end1 - start1

    def step(self, actions):
        self.send_action(actions)

        self._epslen += 1

        agent_obs = self._get_obs()

        done, reward, out_penalty, turning_penalty, locked_penalty, dis_reward, lock_reward = self._get_done_and_reward()

        for i in range(self.n_envs):
            if self._epslen[i] > 1000:
                done[i] = True
                self._draw_rate[i] += 1
                print('max steps')

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
        alive_blue = 1 - (self.alive_units[..., -1] != 1)
        alive_main = 1 - (self.alive_units[..., 0] != 1)

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
            shot_steps=self.shot_steps[i],
            unity_shot_steps=self.unity_shot_steps[i]
        ) for i in range(self.n_envs)]
        agent_reward = [rewards[:, uids] for uids in self.aid2uids]
        agent_discount = [discounts[:, uids] for uids in self.aid2uids]
        agent_reset = [resets[:, uids] for uids in self.aid2uids]

        reset_env = ['0'] * self.n_envs
        for i in range(self.n_envs):
            if done[i]:
                reset_env[i] = '1'
                self._dense_score[i] = np.zeros(self.n_units, np.float32)
                self._turning_penalty[i] = np.zeros(self.n_units, np.float32)
                self._out_penalty[i] = np.zeros(self.n_units, np.float32)
                self._locked_penalty[i] = np.zeros(self.n_units, np.float32)
                self._dis_reward[i] = np.zeros(self.n_units, np.float32)
                self._lock_reward[i] = np.zeros(self.n_units, np.float32)
                self._win_rate[i] = np.zeros(self.n_planes, np.float32)
                self._lose_rate[i] = np.zeros(self.n_planes, np.float32)
                self._draw_rate[i] = np.zeros(self.n_planes, np.float32)
                self.alive_units[i] = np.ones(self.n_planes, np.int32)
                self.locked_warning[i] = np.zeros(self.n_planes, bool)
                self.shot_warning[i] = np.zeros(self.n_planes, bool)
                self.radar_locked[i] = np.zeros(self.n_planes, bool)
                self._epslen[i] = 0
                self.red_shot_time[i] = np.zeros(self.n_red_main, np.int32)
                self.blue_shot_time[i] = np.zeros(self.n_blue_main, np.int32)
                self.prev_action = None
        if '1' in reset_env:
            self.env.reset_envs_with_ids(reset_env)
            self.env.step()

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

    def _get_obs(self, reward=None, action=None):
        obs = [{} for _ in range(self.n_agents)]

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
        plane = plane_infos[0:21]
        weapon = plane_infos[21:25]
        radar = plane_infos[109:]
        # radar = plane_infos[i, 25:33]

        obs = []

        alive = plane[1]
        position = get_xyz(plane_infos)
        angle = get_angle(plane_infos)
        obs.extend(np.concatenate([
            x2tri(np.deg2rad(angle[0])), x2tri(np.deg2rad(angle[1])), x2tri(np.deg2rad(angle[2]))
        ], -1))
        dis_x = np.min([position[0] - EDGE_X[0], EDGE_X[1] - position[0]])
        dis_y = np.min([position[1] - EDGE_Y[0], EDGE_Y[1] - position[1]])
        dis_z = np.min([position[2] - EDGE_Z[0], EDGE_Z[1] - position[2]])
        edge_dis = [dis_x, dis_y, dis_z]
        obs.extend(edge_dis)
        self.alive_units[team][i] = alive if self.alive_units[team][i] != -1 else -1
        vel, vel_scalar = get_velocity(plane_infos)
        obs.extend(vel)
        obs.append(vel_scalar)

        overload = plane[17]
        obs.append(overload)
        # overload = np.clip(overload, -2, 2)

        # oil = plane[18]
        locked_warning = plane[19]
        shot_warning = plane[20]
        obs.extend([locked_warning, shot_warning])
        self.locked_warning[team][i] = locked_warning
        self.shot_warning[team][i] = shot_warning

        # me_missile = self._get_target_missile(blue_weapon, team, i, positions[i])
        # TODO: This should be modified when there are more than two planes
        enemy_dist = self.position[team][1 - side] - self.position[team][side]
        obs.extend(enemy_dist)
        # threshold = 1000
        # oil_warning = 1 if oil < threshold else 0
        # which_missile = weapon[i][0] if i < self.n_red_main else 0
        mid_dis_missile = weapon[1]
        if side == 0:
            self.red_missile_left[team][0][0] = mid_dis_missile
        else:
            self.blue_missile_left[team][0][0] = mid_dis_missile
        # short_dis_missile = weapon[i][2] if i < self.n_red_main else 0
        # disrupt_missile = weapon[i][3] if i < self.n_red_main else 0
        # alive_blue_num = sum(blue_self[:, 1])
        obs.extend(
            self.to_one_hot(int(mid_dis_missile), 5)
        )

        radar_target = self.to_one_hot(int(radar[1]) + 1, 3)
        # self.radar_locked[team][i] = 1 if radar[1] != -1 else 0
        self.radar_locked[team][i] = radar[4]
        obs.extend(radar_target)

        # radar_detect = radar[4]
        radar_detect_yxz = radar[5:8]
        obs.extend(radar_detect_yxz)
        # print(obs)

        # all_action_mask.append(np.ones(self.action_dim[self.uid2aid[i]], bool))

        obs = np.array(obs, np.float32)  # 30
        if not alive:
            obs = np.zeros_like(obs)
        return obs

    def _process_obs(self, prev_reward, prev_action):
        ds_obs = self.decision_steps.obs[0]
        ids = np.argsort(self.decision_steps.agent_id)
        ds_obs = ds_obs[ids].reshape(self.n_envs, self.n_units, 133)
        self.ds_obs = ds_obs
        assert ds_obs.shape == (self.n_envs, self.n_units, 133), ds_obs.shape

        all_env_obs = []
        self.blue_missile_left = np.zeros((self.n_envs, self.n_blue_main, 3), np.int32)
        self.red_missile_left = np.zeros((self.n_envs, self.n_red_main, 3), np.int32)

        positions = self._get_all_positions(ds_obs)
        assert positions.shape == (self.n_envs, self.n_units, 3), positions

        self.last_position = self.position.copy()
        self.position = positions


        # 遍历红方飞机，计算observation
        for team in range(self.n_envs):
            observations = {}
            all_obs, all_global_state, all_alive, all_action_mask, all_prev_action = [], [], [], [], []

            for i in range(self.n_red_main):
                obs = self._construct_obs(ds_obs[team, i, :], team, 0, i)
                global_state = obs

                all_obs.append(obs.copy())
                all_global_state.append(global_state.copy())
                all_alive.append(self.alive_units[team][i])
                prev_action = np.zeros((self.action_dim[self.uid2aid[i]]), np.float32)
                all_prev_action.append(prev_action)

            for i in range(self.n_blue_main):
                i = sum([len(self.aid2uids[k]) for k in range(1)]) + i
                obs = self._construct_obs(ds_obs[team, i, :], team, 1, i)
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
            observations['prev_reward'] = np.zeros(self.n_units, np.float32)
            observations['prev_action'] = all_prev_action

            all_env_obs.append(observations)

        return all_env_obs

    def to_one_hot(self, m, n):
        one_hot = [0] * n
        one_hot[m] = 1
        return one_hot

    def _get_all_positions(self, all_states):
        pos = np.array(all_states[..., 2:5])
        # assert np.all(pos[0] <= EDGE_X[1] + TOLERANCE)
        # assert np.all(pos[0] >= EDGE_X[0] - TOLERANCE)
        # assert np.all(pos[1] <= EDGE_Y[1] + TOLERANCE)
        # assert np.all(pos[1] >= EDGE_Y[0] - TOLERANCE)
        # assert np.all(pos[2] <= EDGE_Z[1] + TOLERANCE)
        # assert np.all(pos[2] >= EDGE_Z[0] - TOLERANCE)
        return pos

    def _get_rel_positions(self, pos):
        rel_pos = np.expand_dims(pos, 0) - np.expand_dims(pos, 1)
        return rel_pos


    def set_actions(self, move_action, radar_action, shot_action):
        move_action = move_action.reshape((self.n_envs * self.n_units, 5))
        agent_ids = self.decision_steps.agent_id
        indexs = np.argsort(self.decision_steps.agent_id)
        action_tuple = self.env.get_action_tuple()

        unsort = np.empty_like(indexs)
        unsort[indexs] = np.arange(indexs.size)
        continue_a = move_action[unsort]
        continue_a = continue_a / 2
        continue_a[..., 4] = (continue_a[..., 4] + 1) / 2
        continue_a = np.concatenate((continue_a, np.zeros((len(agent_ids), 2))), -1)

        # continue_a = np.empty_like(np.zeros((self.n_envs * self.n_units, 7)))
        # for i in range(len(continue_a)):
        #     continue_a[i] = np.array([0, 0, 0, 0.1, 1, 0, 0])

        disc_a = np.concatenate((shot_action,
                                 np.zeros((self.n_envs, self.n_units, 1)),
                                 radar_action,
                                 np.zeros((self.n_envs, self.n_units, 1))
                                 ), -1)
        disc_a = disc_a.reshape((self.n_envs * self.n_units, 4))
        disc_a = disc_a[unsort]
        action_tuple.add_discrete(disc_a)
        action_tuple.add_continuous(continue_a)
        self.env.set_actions(self.name, action_tuple)


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
            # for id, p in enumerate(self.position[i]):
            #     if p[0] > EDGE_X[1] or p[0] < EDGE_X[0] or p[1] > EDGE_Y[1] or p[1] < EDGE_Y[0] or p[2] > EDGE_Z[
            #         1] or p[2] < EDGE_Z[0]:
            #         self.alive_units[i][id] = -1
            #         reward[i][id] += -5
            #         out_penalty[i][id] += -5

            if self.alive_units[i][-1] != 1 and self.alive_units[i][0] == 1:
                done[i] = True
                reward[i][0] += self.reward_config['blue_dead_reward']
                reward[i][1] -= self.reward_config['blue_dead_reward']
                self._win_rate[i] += 1
                print('red win')

            # 红方主机死亡
            if self.alive_units[i][0] != 1 and self.alive_units[i][-1] == 1:
                done[i] = True
                reward[i][0] += self.reward_config['main_dead_reward']
                reward[i][1] -= self.reward_config['main_dead_reward']
                # self.alive_steps[i][0] = self._epslen[i]
                self._lose_rate[i] += 1
                self.alive_units[i][0] = -1
                print('red lose')

            if self.alive_units[i][0] != 1 and self.alive_units[i][-1] != 1:
                done[i] = True
                reward[i][0] += -10
                reward[i][1] += -10
                self._draw_rate[i] += 1
                print('draw')

            if done[i] == 0:

                for j in range(self.n_red_main):
                    if self.radar_locked[i][j] == 1:
                        reward[i][j] += LOCK_REWARD
                        lock_reward[i][j] += LOCK_REWARD

                    if self.locked_warning[i][j] == 0:
                        if abs(self.dis[i][j][0]) > 0.01 or abs(self.dis[i][j][1]) > 0.02 or abs(self.dis[i][j][2]) > 0.02:
                            reward[i][j] -= TURN_PENALTY
                            turning_penalty[i][j] -= TURN_PENALTY
                    else:
                        reward[i][j] -= LOCK_PENALTY
                        locked_penalty[i][j] -= LOCK_PENALTY

                for j in range(self.n_blue_main):
                    if self.radar_locked[i][j + self.n_red_main] == 1:
                        reward[i][j + self.n_red_main] += LOCK_REWARD
                        lock_reward[i][j + self.n_red_main] += LOCK_REWARD

                    if self.locked_warning[i][j + self.n_red_main] == 0:
                        if abs(self.dis[i][j + self.n_red_main][0]) > 0.01 or abs(self.dis[i][j + self.n_red_main][1]) > 0.02 or abs(
                                self.dis[i][j + self.n_red_main][2]) > 0.02:
                            reward[i][j + self.n_red_main] -= TURN_PENALTY
                            turning_penalty[i][j + self.n_red_main] -= TURN_PENALTY
                    else:
                        reward[i][j + self.n_red_main] -= LOCK_PENALTY
                        locked_penalty[i][j + self.n_red_main] -= LOCK_PENALTY

                last_dis = np.linalg.norm(self.last_position[i][0] - self.last_position[i][1])
                now_dis = np.linalg.norm(self.position[i][0] - self.position[i][1])
                if now_dis < last_dis:
                    reward[i] += DIS_REWARD
                    dis_reward += DIS_REWARD

        # if self._epslen[0] != 1 and self.blue_missile_left[0][0][0] == 0 and self.red_missile_left[0][0][0] == 0:
        #     done[i] = True

        self.pre_reward = reward.copy()
        return done, reward, out_penalty, turning_penalty, locked_penalty, dis_reward, lock_reward  # , detect_r


""" Test Code """
if __name__ == '__main__':
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


    from utility import yaml_op

    config = yaml_op.load_config('D:\FlightCombat\CombatTrain\distributed/sync2/configs/unity3_1v1.yaml')
    config = config['env']

    config['fly_control_dir'] = '0826-new-lr=1e-3'
    config['n_envs'] = 1

    env = UnityEnv(**config)
    observations = env.reset()
    print('reset observations')
    # for i, o in enumerate(observations):
    #    print_dict_info(o, f'\tagent{i}')
    all_start = time.time()
    while True:
        # env.env.reset_envs_with_ids([2])
        # actions = env.random_action()
        actions = env.fly_draw_line()
        observations, rewards, dones, infos = env.step(actions)
        # for i, o in enumerate(observations):
        #    print_dict_info(o, f'\tagent{i}')actions = [np.array(([[[0.4, 0, 0.05, 1]]])), np.array(([[[0.4, 0, 0.05, 1]]]))]
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