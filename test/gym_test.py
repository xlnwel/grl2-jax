import random
import numpy as np
import ray

from env.gym_env import create_gym_env
from utility.timer import Timer


default_config = dict(
    name='Pendulum-v0',
    video_path='video',
    log_video=False,
    max_episode_steps=1000,
    clip_reward=None,
    seed=0
)

class TestClass:
    def test_GymEnv(self):
        config = default_config.copy()
        config['n_envs'] = 1
        env = create_gym_env(config)
        d = False
        cr = 0
        n = 0
        s = env.reset()
        while not d:
            n += 1
            a = env.random_action()
            s, r, d, _ = env.step(a)
            cr += r

        assert cr == env.get_score()
        assert n == env.get_epslen()
        return cr, n

    def test_GymEnvVecBase(self):
        config = default_config.copy()
        config['n_envs'] = 64
        env = create_gym_env(config)
        d = False
        cr = np.zeros(env.n_envs)
        n = np.zeros(env.n_envs)
        s = env.reset()
        for _ in range(100):
            a = env.random_action()
            s, r, d, _ = env.step(a)
            cr += np.squeeze(np.where(env.get_mask(), r, 0))
            n += np.squeeze(np.where(env.get_mask(), 1, 0))
        assert np.all(cr == env.get_score()), f'counted reward: {cr}\nrecorded reward: {env.get_score()}'
        assert np.all(n == env.get_epslen()), f'counted epslen: {n}\nrecorded epslen: {env.get_epslen()}'
            
        return cr, n

    def test_GymEnvVec(self):
        ray.init()
        config = default_config.copy()
        config['n_envs'] = n_envs = 8
        config['n_workers'] = 8
        n = 8
        env = create_gym_env(config)
        cr = np.zeros(env.n_envs)
        n = np.zeros(env.n_envs)
        s = env.reset()
        for _ in range(100):
            a = env.random_action()
            s, r, d, _ = env.step(a)
            cr += np.squeeze(np.where(env.get_mask(), r, 0))
            n += np.squeeze(np.where(env.get_mask(), 1, 0))
        assert np.all(cr == env.get_score()), f'counted reward: {cr}\nrecorded reward: {env.get_score()}'
        assert np.all(n == env.get_epslen()), f'counted epslen: {n}\nrecorded epslen: {env.get_epslen()}'

        ray.shutdown()
        return cr
