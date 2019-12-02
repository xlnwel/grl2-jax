import random
import numpy as np
import ray

from env.gym_env import create_gym_env
from utility.timer import Timer


default_config = dict(
    name='BipedalWalkerHardcore-v2',
    video_path='video',
    log_video=False,
    clip_reward=None,
    seed=0
)

class TestClass:
    def test_Env(self):
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

    def test_EnVec(self):
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
        np.testing.assert_allclose(cr, env.get_score())
        np.testing.assert_allclose(n, env.get_epslen())
        assert np.all(n == env.get_epslen()), f'counted epslen: {n}\nrecorded epslen: {env.get_epslen()}'

    def test_EfficientEnvVec(self):
        config = default_config.copy()
        config['n_envs'] = 64
        config['efficient_envvec'] = True
        env = create_gym_env(config)
        d = False
        cr = np.zeros(env.n_envs)
        n = np.zeros(env.n_envs)
        s = env.reset()
        for _ in range(100):
            a = env.random_action()
            s, r, d, info = env.step(a)
            env_ids = np.array([i['env_id'] for i in info])
            cr[env_ids] += r
            n[env_ids] += 1
        np.testing.assert_allclose(cr, env.get_score(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(n, env.get_epslen())
        assert np.all(n == env.get_epslen()), f'counted epslen: {n}\nrecorded epslen: {env.get_epslen()}'

    # def test_RayEnvVec(self):
    #     ray.init()
    #     config = default_config.copy()
    #     config['n_envs'] = 8
    #     config['n_workers'] = 8
    #     n = 8
    #     env = create_gym_env(config)
    #     cr = np.zeros(env.n_envs)
    #     n = np.zeros(env.n_envs)
    #     s = env.reset()
    #     for _ in range(100):
    #         a = env.random_action()
    #         s, r, d, _ = env.step(a)
    #         cr += np.squeeze(np.where(env.get_mask(), r, 0))
    #         n += np.squeeze(np.where(env.get_mask(), 1, 0))
    #     np.testing.assert_allclose(cr, env.get_score())
    #     assert np.all(n == env.get_epslen()), f'counted epslen: {n}\nrecorded epslen: {env.get_epslen()}'

    #     ray.shutdown()

    def test_atari(self):
        config = default_config.copy()
        config['is_deepmind_env'] = True
        config['name'] = 'BreakoutNoFrameskip-v4'
        env = create_gym_env(config)
        d = False
        cr = 0
        n = 0
        s = env.reset()
        while not env.env.already_done:
            n += 1
            a = env.random_action()
            s, r, d, _ = env.step(a)
            cr += r

        assert cr == env.get_score()
        assert n == env.get_epslen()
