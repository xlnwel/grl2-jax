import platform
import math
import time

import numpy as np
import gym
from .interface import UnityInterface
from core.tf_config import configure_gpu

"""  Naming Conventions:
An Agent is the one that makes decisions, it's typically a RL agent
A Unit is a controllable unit in the environment.
Each agent may control a number of units.
All ids start from 0 and increases sequentially
"""

MAX_V = 0.544  # max 1.3 min 0.7
MIN_V = 0.238
THETA_RANGE = [-1, 1]
PHI_RANGE = [-0.08, 0.08]
ROLL_RANGE = [-0.5, 0.5]
END_THRESHOLD = [0.01, 0.02, 0.02, 0.02]
LOW_HEIGHT = 2
BOMB_PENALTY = -10
SUCCESS_REWARD = 10
STEP_PENALTY = 0.002
DELTA_V = [0.2, 1, 0.5, 0.5]
DRAW_LINE = False

TOLERANCE = 1000

MAX_SPEED = 700
MIN_SPEED = 10
MAX_ANGLE_SPEED = 12
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
    vel_scalar = np.linalg.norm(vel)
    assert np.all(vel_scalar < MAX_SPEED), vel
    assert np.all(vel_scalar > MIN_SPEED), vel
    return vel / 1000, vel_scalar / 1000


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
    assert 1500 < h < 21000, h
    return h / 1000


def get_xyz(state):
    xyz = state[..., 2:5]
    return xyz / 1000


def get_overload(state):
    o = state[..., 14:15]
    # assert -3 < o < 9, o
    return o


# NOTE: Keep the class name fixed; do not invent a new one!
# We do not rely on this for distinction!
class UnityEnv:
    def __init__(
            self,
            uid2aid,
            n_envs,
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
            # expand kwargs for your environment
            **kwargs
    ):
        # uid2aid is a list whose indices are the unit ids and values are agent ids.
        # It specifies which agent controlls the unit.
        # We expect it to be sorted in the consecutive ascending order
        # That is, [0, 1, 1] is valid. [0, 1, 0] and [0, 0, 2] are invalid
        configure_gpu(None)

        self.uid2aid: list = uid2aid
        self.aid2uids = compute_aid2uids(self.uid2aid)
        self.n_agents = len(self.aid2uids)  # the number of agents
        self.n_units = len(self.uid2aid)
        self.frame_skip = 10
        self.unity_config = unity_config
        if platform.system() == 'Windows':
            # self.unity_config['file_name'] = 'D:/FlightCombat/fly_win/T2.exe'
            # self.unity_config['worker_id'] = 100
            self.unity_config['file_name'] = None
            self.unity_config['worker_id'] = 0
        #        else:
        #            self.unity_config['file_name'] = '/home/ubuntu/wuyunkun/hm/env/unity_env/data/blue_fly/3d.x86_64'
        #            self.unity_config['worker_id'] = 10

        # unity_config['n_envs'] = n_envs
        self.env = UnityInterface(**self.unity_config)
        # the number of envs running in parallel, which must be the same as the number
        # of environment copies specified when compiing the Unity environment
        self.n_envs = n_envs
        # The same story goes with n_units
        self._seed = np.random.randint(1000) if seed is None else seed
        # The maximum number of steps per episode;
        # the length of an episode should never exceed this value
        self.max_episode_steps = kwargs['max_episode_steps']
        self.reward_config = reward_config
        self.use_action_mask = False  # if action mask is used
        self.use_life_mask = False  # if life mask is used

        self._unity_disc_actions = 2
        self._unity_cont_actions = 2
        # self.is_action_discrete = is_action_discrete
        # if is_action_discrete:
        #    self.action_dim = [3]
        #    self.action_space = [gym.spaces.Discrete(ad) for ad in self.action_dim]
        # else:
        self.action_dim = [5]
        self.action_space = [gym.spaces.Box(low=-1, high=1, shape=(ad,)) for ad in self.action_dim]
        self._obs_dim = [15]
        self._global_state_dim = [15]
        self.is_multi_agent = False
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

        # The length of the episode
        self._epslen = np.zeros(self.n_envs, np.int32)

        self.pre_reward = np.zeros((self.n_envs, self.n_units), np.float32)
        self.prev_action = None
        self._target_vel = np.array((self.n_envs, 4), np.float32)
        self.name = 'E0_Red_0?team=0'
        self.last_v = np.array((self.n_envs, 4), np.float32)
        self._v = np.array((self.n_envs, 4), np.float32)
        self._info = {}
        self._height = 6
        self._dense_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self.draw_target = [[0.4, 0.3, 0.05, 0], [0.25, -0.3, -0.05, 0], [0.333, 0, 0, 0], [0.333, -1, 0, 0]]

    def random_action(self):
        actions = []
        for aid, uids in enumerate(self.aid2uids):
            a = np.array([[self.action_space[aid].sample() for _ in uids] for _ in range(self.n_envs)], np.float32)
            actions.append(a)
        return actions

    def reset(self):
        self._epslen = np.zeros(self.n_envs, np.int32)
        self.last_v = np.zeros((self.n_envs, 4), np.float32)
        self._v = np.zeros((self.n_envs, 4), np.float32)

        self._target_vel = np.zeros((self.n_envs, 4))

        self._generate_target_velocity()
        self._dense_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        done, decision_steps, terminal_steps = self.env.step()
        obs = self.get_obs(decision_steps)
        self.last_v[0] = self._v[0]

        self.pre_reward = np.zeros((self.n_envs, self.n_units), np.float32)
        self.prev_action = None

        return obs

    def _generate_target_velocity(self):
        if DRAW_LINE:
            time.sleep(5)
            i = np.random.randint(4)
            self._target_vel[0] = np.array(self.draw_target[i])
            # del self.draw_target[0]
        else:
            if self._height <= 2.5:
                target_phi = np.random.uniform(0.03, PHI_RANGE[1])
            else:
                target_phi = np.random.uniform(0.03, PHI_RANGE[1]) * np.random.choice([-1, 1])

            target_v = np.random.uniform(MIN_V, MAX_V)

            target_theta = np.random.uniform(0.03, THETA_RANGE[1]) * np.random.choice([-1, 1])
            target_roll = np.random.uniform(0.0, ROLL_RANGE[1]) * np.random.choice([-1, 1])

            self._target_vel[0] = np.hstack((target_v, target_theta, target_phi, target_roll))

    def step(self, action):
        for i in range(self.frame_skip):
            self.set_actions(action, i)
            reset, decision_steps, terminal_steps = self.env.step()
            if reset:
                break
        self._epslen[0] += 1

        agent_obs = self.get_obs(decision_steps, self.pre_reward)

        done, reward, score, fail, edge = self._get_done_and_reward()

        discount = 1 - done  # we return discount instead of done

        assert reward.shape == (self.n_envs, self.n_units), reward.shape
        assert discount.shape == (self.n_envs,), discount.shape

        rewards = reward
        discounts = np.tile(discount.reshape(-1, 1), (1, self.n_units))

        self._dense_score += rewards

        reset = done
        assert reset.shape == (self.n_envs,), reset.shape
        resets = np.tile(reset.reshape(-1, 1), (1, self.n_units))

        self.prev_action = action

        self._info = [dict(
            score=np.array([score[i] * self.n_units]),
            is_success=np.array([score[i] * self.n_units]),
            crash=np.array([fail[i] * self.n_units]),
            dense_score=self._dense_score[i].copy(),
            epslen=np.array([self._epslen[i]] * self.n_units),
            game_over=np.array([discount[i] == 0] * self.n_units),
            obs_target_v0=np.array([self._target_vel[0][0]] * self.n_units),
            obs_target_v1=np.array([self._target_vel[0][1]] * self.n_units),
            obs_target_v2=np.array([self._target_vel[0][2]] * self.n_units),
            obs_target_v3=np.array([self._target_vel[0][3]] * self.n_units),
            obs_now_v0=np.array([self._v[0][0]] * self.n_units),
            obs_now_v1=np.array([self._v[0][1]] * self.n_units),
            obs_now_v2=np.array([self._v[0][2]] * self.n_units),
            obs_now_v3=np.array([self._v[0][3]] * self.n_units)
            # obs_now_height=np.array(np.array([self._height]) * self.n_units)
            # obs_overload=np.array([agent_obs[0]['obs'][0][0][14]] * self.n_units),
        ) for i in range(self.n_envs)]
        agent_reward = [rewards[:, uids] for uids in self.aid2uids]
        agent_discount = [discounts[:, uids] for uids in self.aid2uids]
        agent_reset = [resets[:, uids] for uids in self.aid2uids]

        for i in range(self.n_envs):
            if done[i]:
                if fail[i] or edge[i]:
                    self.env.reset()
                self.reset()

        return agent_obs, agent_reward, agent_discount, agent_reset

    def info(self):
        return self._info

    def close(self):
        # close the environment
        pass

    def _get_obs_shape(self, aid):
        return (self._obs_dim[aid],)

    def _get_global_state_shape(self, aid):
        return (self._global_state_dim[aid],)

    def get_obs(self, d_s, reward=None, action=None):
        ds = d_s[self.name].obs[0][0]
        _, v_scalar = get_velocity(ds)
        angle_v = get_angle_velocity(ds)
        angle = get_angle(ds)

        theta = angle[1]
        phi = angle[0]
        roll = angle[2]

        now_v = np.stack([v_scalar, theta / 180, phi / 180, roll / 180], -1)
        dis = self._target_vel[0] - now_v
        dis[..., 1:] = dis[..., 1:] - np.copysign(dis[..., 1:], 2)
        # assert dis.shape == (self.n_envs, self.n_units, 4), dis.shape

        v_ = np.expand_dims(v_scalar, -1)

        posture = np.concatenate([
            x2tri(np.deg2rad(theta)) + x2tri(np.deg2rad(phi)) + x2tri(np.deg2rad(roll))
        ], -1)
        # assert posture.shape == (self.n_envs, self.n_units, 6), dis.shape
        height = get_height(ds)
        # assert height.shape == (self.n_envs, self.n_units, 1), height.shape
        obs = np.concatenate((
            v_,  # 1
            angle_v,  # 3
            posture,  # 6
            height,  # 1
            dis,  # 4
        ), -1)
        observations = {}
        observations['obs'] = obs
        observations['global_state'] = obs
        observations['prev_reward'] = np.zeros(self.n_units, np.float32)
        observations['prev_action'] = np.zeros((self.action_dim[self.uid2aid[0]]), np.float32)
        all_obs = [observations]

        self._v[0] = now_v.copy()
        self.xyz = get_xyz(ds)

        r_obs = [{} for _ in range(self.n_agents)]

        for aid in range(self.n_agents):
            for k in self.obs_shape[aid].keys():
                r_obs[aid][k] = np.zeros(
                    (self.n_envs, len(self.aid2uids[aid]), *self.obs_shape[aid][k]),
                    dtype=self.obs_dtype[aid][k]
                )
                r_obs[aid][k][0] = all_obs[0][k]

        return r_obs

    def set_actions(self, action, i):
        action_tuple = self.env.get_action_tuple()
        action_tuple.add_discrete(np.zeros((1, 4)))
        a = action[0][0]

        a[0][4] = (a[0][4] + 1) / 2
        # for j in range(3):
        #     # a[0][j] = (i + 1) * a[0][j] / 10
        #     a[0][j] = a[0][j] / 2
        # if np.abs(a[0][i]) > 0.8:
        #     a[0][i] = a[0][i] - 0.2
        action = np.hstack((a.reshape((1, 5)), np.zeros((1, 2))))
        action_tuple.add_continuous(action)
        self.env.set_actions(self.name, action_tuple)

    def _get_done_and_reward(self):
        """  获取游戏逻辑上done和reward
                """
        done = np.array([False] * self.n_envs, np.float32)
        reward = -1 * STEP_PENALTY * np.ones((self.n_envs, self.n_units), np.float32)
        score = np.zeros(self.n_envs)
        fail = np.zeros(self.n_envs)
        edge = np.zeros(self.n_envs)

        v = self._v[0]

        if self.xyz[0] > 70 or self.xyz[0] < -70 or \
                self.xyz[2] > 80 or self.xyz[2] < -80:
            done[0] = True
            edge[0] = 1
            return done, reward, score, fail, edge

        if self.xyz[1] <= LOW_HEIGHT or self.xyz[1] > 15:
            done[0] = True
            reward[0] += BOMB_PENALTY
            fail[0] = 1
            # print('crash')
            return done, reward, score, fail, edge

        if self._epslen[0] > self.max_episode_steps:
            done[0] = True
            # print('max')

            return done, reward, score, fail, edge

        dis_theta = abs(v[1] - self._target_vel[0][1])
        if dis_theta >= 1:
            dis_theta = 2 - dis_theta

        dis_phi = abs(v[2] - self._target_vel[0][2])
        if dis_phi >= 1:
            dis_phi = 2 - dis_phi

        dis_roll = abs(v[3])

        if dis_theta < END_THRESHOLD[1] and \
                abs(v[0] - self._target_vel[0][0]) < END_THRESHOLD[0] and \
                dis_phi < END_THRESHOLD[2] and dis_roll < END_THRESHOLD[3]:
            done[0] = True
            score[0] = 1
            reward[0] += SUCCESS_REWARD
            # print('success')

            return done, reward, score, fail, edge

        last_v = self.last_v[0]
        delta_theta_t = abs(last_v[1] - self._target_vel[0][1])
        if delta_theta_t >= 1:
            delta_theta_t = 2 - delta_theta_t

        delta_theta_t1 = abs(v[1] - self._target_vel[0][1])
        if delta_theta_t1 >= 1:
            delta_theta_t1 = 2 - delta_theta_t1

        delta_phi_t = abs(last_v[2] - self._target_vel[0][2])
        if delta_phi_t >= 1:
            delta_phi_t = 2 - delta_phi_t

        delta_phi_t1 = abs(v[2] - self._target_vel[0][2])
        if delta_phi_t1 >= 1:
            delta_phi_t1 = 2 - delta_phi_t1

        delta_roll_t = abs(last_v[3])
        delta_roll_t1 = abs(v[3])

        reward[0] += (delta_theta_t - delta_theta_t1) / DELTA_V[1] \
                     + (abs(last_v[0] - self._target_vel[0][0]) - abs(v[0] - self._target_vel[0][0])) / DELTA_V[0] \
                     + (delta_phi_t - delta_phi_t1) / DELTA_V[2] \
                     + (delta_roll_t - delta_roll_t1) / DELTA_V[3]
        # print(reward)
        self.last_v = self._v.copy()

        if v[0] < MIN_V or v[0] > MAX_V:
            reward[0] -= STEP_PENALTY

        self.pre_reward = reward.copy()

        return done, reward, score, fail, edge

    def seed(self, seed):
        """Returns the random seed used by the environment."""
        self._seed = seed


""" Test Code """
if __name__ == '__main__':
    config = dict(
        env_name='dummy',
        uid2aid=[0],
        max_episode_steps=500,
        n_envs=1,
        unity_config={
            'worker_id': 1000,
            'file_name': '/home/ubuntu/wuyunkun/hm/env/unity_env/data/red_fly/3d.x86_64'
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
    for i, o in enumerate(observations):
        print_dict_info(o, f'\tagent{i}')

    for k in range(1, 50000):
        # env.env.reset_envs_with_ids([2])
        actions = env.random_action()
        print(f'Step {k}, random actions', actions)
        observations, rewards, dones, reset = env.step(actions)
        print(f'Step {k}, observations')
        for i, o in enumerate(observations):
            print_dict_info(o, f'\tagent{i}')
        print(f'Step {k}, rewards', rewards)
        print(f'Step {k}, dones', dones)
        print(f'Step {k}, reset', reset)
        info = env.info()
        print(f'Step {k}, info')
        for aid, i in enumerate(info):
            print_dict(i, f'\tenv{aid}')
