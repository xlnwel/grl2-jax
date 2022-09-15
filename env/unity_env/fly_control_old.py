import platform
import math
import time

import numpy as np
import gym
from env.unity_env.interface import UnityInterface
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
        self.frame_skip = 1
        self.unity_config = unity_config
        if platform.system() == 'Windows':
            # self.unity_config['file_name'] = 'D:/FlightCombat/fly_win/T2.exe'
            # self.unity_config['worker_id'] = 100
            self.unity_config['file_name'] = None
            self.unity_config['worker_id'] = 0

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
        self.name = 'Player?team=0'
        self.last_angle = np.array((self.n_envs, 3), np.float32)
        self.last_v = np.array((self.n_envs, 4), np.float32)
        self._v = np.array((self.n_envs, 4), np.float32)
        self._angle = np.array((self.n_envs, 3), np.float32)
        self._info = {}
        self._height = 6
        self._dense_score = np.zeros((self.n_envs, self.n_units), dtype=np.float32)
        self.draw_target = [[0.4, 0.3, 0.05, 0.2], [0.25, -0.3, -0.05, 0.2], [0.333, 0, 0, 0.2]]

    def random_action(self):
        actions = []
        for aid, uids in enumerate(self.aid2uids):
            # a = np.array([[self.action_space[aid].sample() for _ in uids] for _ in range(self.n_envs)], np.float32)
            a = np.array([[[0, 0, 1, 0.5, 1] for _ in uids] for _ in range(self.n_envs)], np.float32)

            actions.append(a)
        return actions

    def reset(self):
        self._epslen = np.zeros(self.n_envs, np.int32)
        self.last_v = np.zeros((self.n_envs, 4), np.float32)
        self.last_angle = np.zeros((self.n_envs, 3), np.float32)
        self._v = np.zeros((self.n_envs, 4), np.float32)
        self._angle = np.zeros((self.n_envs, 3), np.float32)

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
        if True:
            time.sleep(5)
            i = np.random.randint(4)
            self._target_vel[0] = np.array(self.draw_target[i])
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
            self.set_actions(action)
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
            # obs_overload=np.array([agent_obs[0]['obs'][0][0][14]] * self.n_units),
        ) for i in range(self.n_envs)]
        agent_reward = [rewards[:, uids] for uids in self.aid2uids]
        agent_discount = [discounts[:, uids] for uids in self.aid2uids]
        agent_reset = [resets[:, uids] for uids in self.aid2uids]

        for i in range(self.n_envs):
            if done[i]:
                if fail[i] or edge[i]:
                    self.env.reset()
                    self.env.step()

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

    def get_obs(self, ds, reward=None, action=None):
        obs = [{} for _ in range(self.n_agents)]

        all_states = ds[self.name].obs[0][0]
        vel = all_states[5:8] / 1000
        #print(vel)
        v_scalar = np.linalg.norm(vel)
        angle_v = all_states[8:11]
        angle = all_states[11:14]
        print('angle_v:', angle_v)
        print('angle:', angle)

        for i in range(3):
            while abs(angle[i]) > 180:
                angle[i] = angle[i] - math.copysign(360, angle[i])

        theta = angle[1] / 180
        phi = angle[0] / 180
        roll = angle[2] / 180
        v = np.array([v_scalar, theta, phi, roll])
        # print(angle)
        v_ = np.array([v_scalar, phi, roll])
        #v_ = np.array([v_scalar, phi])

        posture = np.concatenate((np.sin(np.deg2rad([angle[0], angle[2]])), np.cos(np.deg2rad([angle[0], angle[2]]))))
        height = all_states[3] / 1000
        np.set_printoptions(suppress=True)
        overload = all_states[14]
        self._angle[0] = angle.copy()
        self._v[0] = v.copy()
        self._height = height
        self.xyz = all_states[2:5] / 1000

        self.overload = np.clip(overload, -2, 2)
        # print(v)
        dis = self._target_vel[0] - v
        if abs(dis[1]) > 1:
            dis[1] = dis[1] - math.copysign(2, dis[1])
        if abs(dis[2]) > 1:
            dis[2] = dis[2] - math.copysign(2, dis[2])

        one_obs = np.hstack((
            v_,
            angle_v,
            posture,
            height / 20,
            dis,
            # self.overload / 2,
            # oil
        ))
        # print('obs:' + str(one_obs))
        observations = {}
        observations['obs'] = one_obs
        observations['global_state'] = one_obs
        observations['life_mask'] = [1]
        mask = [1, 1, 1]
        if v[0] < MIN_V:
            mask = [0, 1, 1]
        if v[0] > MAX_V:
            mask = [1, 1, 0]
        observations['action_mask'] = mask
        observations['prev_reward'] = np.zeros(self.n_units, np.float32)
        observations['prev_action'] = np.zeros((self.action_dim[self.uid2aid[0]]), np.float32)
        all_obs = [observations]

        for aid in range(self.n_agents):
            for k in self.obs_shape[aid].keys():
                obs[aid][k] = np.zeros(
                    (self.n_envs, len(self.aid2uids[aid]), *self.obs_shape[aid][k]),
                    dtype=self.obs_dtype[aid][k]
                )
                obs[aid][k][0] = all_obs[0][k]

        return obs

    def set_actions(self, action):
        action_tuple = self.env.get_action_tuple()
        action_tuple.add_discrete(np.zeros((1, 4)))
        a = action[0][0]
        a[0][4] = (a[0][4] + 1)/2
        action = np.hstack((a.reshape((1, 5)), np.zeros((1, 2))))
        action_tuple.add_continuous(action)
        self.env.set_actions(self.name, action_tuple)

    def _get_done_and_reward(self):
        """  获取游戏逻辑上done和reward
                """
        done = np.array([False] * self.n_envs)
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
        if self._height <= LOW_HEIGHT or self.overload > 9 or self.overload < -3 or self.xyz[1] > 15:
            done[0] = True
            reward[0] += BOMB_PENALTY
            fail[0] = 1
            return done, reward, score, fail, edge

        if self._epslen[0] > self.max_episode_steps:
            done[0] = True
            print('max')

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
            print('success')

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
        self.last_angle = self._angle.copy()
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
        # print(f'Step {k}, random actions', actions)
        print(f'Step {k}')
        observations, rewards, dones, reset = env.step(actions)
        # print(f'Step {k}, observations')
        # for i, o in enumerate(observations):
        #     print_dict_info(o, f'\tagent{i}')
        # print(f'Step {k}, rewards', rewards)
        # print(f'Step {k}, dones', dones)
        # print(f'Step {k}, reset', reset)
        # info = env.info()
        # print(f'Step {k}, info')
        # for aid, i in enumerate(info):
        #     print_dict(i, f'\tenv{aid}')
