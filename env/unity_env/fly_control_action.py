import copy
import random
import numpy
import numpy as np
import gym
from interface import UnityInterface
import math
import time

MAX_V = 0.442  # max 1.3 min 0.7
MIN_V = 0.238
THETA_RANGE = [-1, 1]
PHI_RANGE = [-0.08, 0.08]
END_THRESHOLD = [0.01, 0.02, 0.02]
LOW_HEIGHT = 2
BOMB_PENALTY = -10
SUCCESS_REWARD = 10
STEP_PENALTY = 0.002
DELTA_V = [0.2, 1, 0.5]


# OIL_THRESHOLD = 1000


# NOTE: Keep the class name fixed; do not invent a new one!
# We do not rely on this for distinction!
class UnityEnvEval(gym.Env):
    def __init__(
            self,
            n_envs=1,
            unity_config={
                'file_name': '/home/ubuntu/wuyunkun/hm/run/test_baselines/unity/3d.x86_64'
            },
            frame_skip=5,
            reward_config={
            },
            # expand kwargs for your environment
            **kwargs
    ):
        self.frame_skip = 1
        unity_config['worker_id'] = 0
        # unity_config['worker_id'] = np.random.randint(0, 10000)
        # unity_config['n_envs'] = n_envs
        self.env = UnityInterface(**unity_config)
        self.n_envs = 1

        self.max_episode_steps = 500
        self.reward_config = reward_config
        self._action_dim = 1
        self._obs_dim = 15
        self.reward_range = (0, 1)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self._obs_dim,))
        # self.action_space = gym.spaces.Discrete(3)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self._action_dim,))
        self.metadata = {
            'render.modes': [],
        }
        self._epslen = np.zeros(self.n_envs, np.int32)
        self._init_vel = np.array((self.n_envs, 3), np.float32)
        self._target_vel = np.array((self.n_envs, 3), np.float32)
        self.name = 'E0_Red_0?team=0'
        self.last_angle = np.array((self.n_envs, 3), np.float32)
        self.last_v = np.array((self.n_envs, 3), np.float32)
        self.alive = np.array((self.n_envs, 1), np.int32)
        self._v = np.array((self.n_envs, 3), np.float32)
        self._angle = np.array((self.n_envs, 3), np.float32)
        self._info = {}
        self._height = 6
        #self.env.reset()

    def reset(self):
        #self.step_zero()
        ds, ts = self.env.reset()
        self.get_first_obs(ds)

    def get_first_obs(self, ds):
        all_states = ds[self.name].obs[0][0]
        #all_states = np.around(all_states, 3)
        self.alive[0] = all_states[1]
        vel = all_states[5:8]
        v_scalar = np.linalg.norm(vel)
        angle = all_states[8:11]
        print('angle: '+str(angle))
        print('vel: '+str(vel))
        print('vel scalar: '+str(v_scalar))

    def step(self, action):
        action_tuple = self.env.get_action_tuple()
        dis_a = np.zeros((1, 4))
        dis_a[0][0] = 1
        dis_a[0][2] = 1
        action_tuple.add_discrete(dis_a)
        action = np.hstack((action.reshape((1, 5)), np.zeros((1, 2))))
        action_tuple.add_continuous(action)
        self.env.set_actions(self.name, action_tuple)
        #self.env.set_actions('E0_Blue_0?team=0', action_tuple)
        done, ds, ts = self.env.step()
        self.get_first_obs(ds)


if __name__ == '__main__':
    start = (time.time())

    config = dict(
        env_name='dummy',
        max_episode_steps=20000,
        n_envs=1,
        unity_config={
            #'worker_id': 1,
            #'file_name': '/home/ubuntu/wuyunkun/hm/run/test_baselines/unity/3d.x86_64'
        },
        reward_config={
        }
    )


    n_unity_file = 1
    n_unity_env = []

    for n in range(n_unity_file):
        n_unity_env.append(UnityEnvEval(**config))
        # config['unity_config']['worker_id'] = config['unity_config']['worker_id'] + 1

    # assert False
    env = n_unity_env[0]
    obs = env.reset()

    print('reset observations')
    game = 0
    success = 0
    while game < 1:
        step = 0

        #print('game {}'.format(game))

        # while step < 500:
        #     print('game {}, step {} time {}'.format(game, step, time.time()))
        #     env.step(np.array([1, 0, 0, 0]))
        #     #print(time.time())
        #     step += 1
        # while step < 90:
        #     print('game {}, step {} time {}'.format(game, step, time.time()))
        #     env.step(np.array([0, 0, -1, 0]))
        #     #print(time.time())
        #     step += 1
        # while step < 270:
        #     print('game {}, step {} time {}'.format(game, step, time.time()))
        #     env.step(np.array([1, 0, 0, 0]))
        #     # print(time.time())
        #     step += 1
        # while step < 360:
        #     print('game {}, step {} time {}'.format(game, step, time.time()))
        #     env.step(np.array([0, 0, 1, 0]))
        #     # print(time.time())
        #     step += 1
        while step < 200:
            print('game {}, step {} time {}'.format(game, step, time.time()))
            env.step(np.array([0, 0, 0.1, 0.5, 1]))
            #env.step(np.random.uniform(-1, 1, (1, 4)))
            #print(time.time())
            step += 1
        env.env.reset()
        step = 0
        while step < 100:
            print('game {}, step {} time {}'.format(game, step, time.time()))
            env.step(np.array([0, 0, 0, 0.5, 1]))
            #env.step(np.random.uniform(-1, 1, (1, 4)))
            #print(time.time())
            step += 1
        game += 1
    print(time.time() - start)
