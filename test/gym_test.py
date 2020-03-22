import random
import numpy as np
import ray

from env.gym_env import create_env
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
        env = create_env(config)
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
        config['n_envs'] = 5
        env = create_env(config)
        d = False
        cr = np.zeros(env.n_envs)
        n = np.zeros(env.n_envs)
        s = env.reset()
        for _ in range(100):
            a = env.random_action()
            s, r, d, _ = env.step(a)
            cr += np.squeeze(np.where(env.get_mask(), r, 0))
            n += np.squeeze(np.where(env.get_mask(), 1, 0))
        np.testing.assert_allclose(cr, env.get_score(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(n, env.get_epslen(), rtol=1e-5, atol=1e-5)

    def test_EfficientEnvVec(self):
        config = default_config.copy()
        config['n_envs'] = 5
        config['efficient_envvec'] = True
        env = create_env(config)
        d = False
        cr = np.zeros(env.n_envs)
        n = np.zeros(env.n_envs)
        s = env.reset()
        for _ in range(1000):
            a = env.random_action()
            s, r, d, info = env.step(a)
            env_ids = np.array([i['env_id'] for i in info])
            cr[env_ids] += r
            n[env_ids] += 1
            if np.all(d):
                break
        np.testing.assert_allclose(cr, env.get_score(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(n, env.get_epslen(), rtol=1e-5, atol=1e-5)
        assert np.all(n == env.get_epslen()), f'counted epslen: {n}\nrecorded epslen: {env.get_epslen()}'

    def test_RayEnvVec(self):
        ray.init()
        config = default_config.copy()
        config['n_envs'] = 2
        config['n_workers'] = 2
        n = 8
        env = create_env(config)
        cr = np.zeros(env.n_envs)
        n = np.zeros(env.n_envs)
        s = env.reset()
        for _ in range(100):
            a = env.random_action()
            s, r, d, _ = env.step(a)
            cr += np.squeeze(np.where(env.get_mask(), r, 0))
            n += np.squeeze(np.where(env.get_mask(), 1, 0))
        np.testing.assert_allclose(cr, env.get_score())
        assert np.all(n == env.get_epslen()), f'counted epslen: {n}\nrecorded epslen: {env.get_epslen()}'

        ray.shutdown()

    def test_atari(self):
        config = default_config.copy()
        config['is_deepmind_env'] = True
        config['name'] = 'BreakoutNoFrameskip-v4'
        env = create_env(config)
        d = False
        cr = 0
        n = 0
        s = env.reset()
        while not env.env.was_real_done:
            assert env.env.lives > 0, f'step({n}), lives({env.env.lives})'
            a = env.random_action()
            s, r, d, _ = env.step(a)
            cr += r
            n += 1
            if d and not env.env.was_real_done:
                assert n == env.get_epslen()
                s = env.reset()
                
                assert n == env.get_epslen()
            assert n == env.get_epslen()
        assert env.env.lives == 0
        assert cr == env.get_score()
        assert n == env.get_epslen()

    def test_mask(self):
        config = default_config.copy()
        for name in ['BipedalWalkerHardcore-v2', 'BreakoutNoFrameskip-v4']:
            config['name'] = name
            env = create_env(config)
            
            has_done = False
            s = env.reset()
            d = False
            while not d:
                a = env.random_action()
                s, r, d, _ = env.step(a)
                
                assert env.get_mask() == 1
                
            for _ in range(10):
                a = env.random_action()
                s, r, d, _ = env.step(a)
                assert env.get_mask() == 0

    def test_auto_reset(self):
        config = default_config.copy()
        config['auto_reset'] = True
        for name in ['BipedalWalkerHardcore-v2', 'BreakoutNoFrameskip-v4']:
            config['name'] = name
            env = create_env(config)
            
            has_done = False
            s = env.reset()
            d = False
            for _ in range(10000):
                a = env.random_action()
                s, r, d, _ = env.step(a)
                assert env.get_mask() == 1

    def test_action_repetition(self):
        ray.init()
        config = default_config.copy()
        config['action_repetition'] = True
        for n_workers in [1, 2]:
            for n_envs in [1, 2]:
                config['n_workers'] = n_workers
                config['n_envs'] = n_envs
                for name in ['BipedalWalkerHardcore-v2', 'BreakoutNoFrameskip-v4']:
                    config['name'] = name
                    env = create_env(config)
                    
                    has_done = False
                    s = env.reset()
                    cr = np.zeros(env.n_envs)
                    n = np.zeros(env.n_envs)
                    for _ in range(1000):
                        n_ar = np.random.randint(1, 10, size=(n_workers, n_envs))
                        n_ar = np.squeeze(n_ar)
                        a = env.random_action()
                        s, r, d, info = env.step(a, n_ar=n_ar)
                        cr += np.where(env.get_mask(), r, 0)
                        if env.n_envs == 1:
                            n += info['n_ar']
                        else:
                            n_ = np.array([i['n_ar'] for i in info])
                            n += np.where(env.get_mask(), n_, 0)
                            
                        np.testing.assert_equal(env.get_epslen(), n)
                        np.testing.assert_allclose(env.get_score(), cr, atol=1e-5, rtol=1e-5)
                        if np.all(d):
                            break
            
        ray.shutdown()
