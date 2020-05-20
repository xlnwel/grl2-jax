""" Implementation of single process environment """
import itertools
import numpy as np
import gym
import tensorflow as tf
import ray
import cv2

from utility.utils import isscalar, RunningMeanStd
from env.wrappers import *
from env.atari import make_atari_env


def make_env(config):
    config = config.copy()
    if 'atari' in config['name'].lower():
        # for atari games, we expect to following name convention 'atari_name'
        _, config['name'] = config['name'].split('_', 1)
        config['max_episode_steps'] = max_episode_steps = 108000    # 30min
        env = make_atari_env(config)
    else:
        env = gym.make(config['name']).env
        env = Dummy(env)
        max_episode_steps = config.get('max_episode_steps', env.spec.max_episode_steps)
        if config.get('frame_skip'):
            env = FrameSkip(env, config['frame_skip'])
    env = EnvStats(env, max_episode_steps,
                    precision=config.get('precision', 32), 
                    timeout_done=config.get('timeout_done', False))
    if 'reward_scale' in config or 'reward_clip' in config:
        env = RewardHack(env, **config)
    if config.get('log_episode'):
        env = LogEpisode(env)

    return env

def create_env(config, env_fn=make_env, force_envvec=False):
    if config.get('n_workers', 1) <= 1:
        EnvType = EnvVec if force_envvec or config.get('n_envs', 1) > 1 else Env
        if EnvType == EnvVec and config.get('efficient_envvec', False):
            EnvType = EfficientEnvVec
        env = EnvType(config, env_fn)
    else:
        EnvType = EnvVec if config.get('n_envs', 1) > 1 else Env
        env = RayEnvVec(EnvType, config, env_fn)

    return env


class EnvVecBase(Wrapper):
    def _convert_batch_obs(self, obs):
        if isinstance(obs[0], np.ndarray):
            obs = np.reshape(obs, [-1, *self.obs_shape])
        else:
            obs = list(obs)
        return obs

    def _get_idxes(self, idxes):
        idxes = idxes or list(range(self.n_envs))
        if isinstance(idxes, int):
            idxes = [idxes]
        return idxes


class Env(Wrapper):
    def __init__(self, config, env_fn=make_env):
        self.env = env_fn(config)
        if 'seed' in config and hasattr(self.env, 'seed'):
            self.env.seed(config['seed'])
        self.name = config['name']
        self.max_episode_steps = self.env.max_episode_steps

    def reset(self, idxes=None, **kwargs):
        return self.env.reset(**kwargs)

    @property
    def n_envs(self):
        return 1

    def random_action(self, *args, **kwargs):
        action = self.env.action_space.sample()
        return action
        
    def step(self, action, **kwargs):
        return self.env.step(action, **kwargs)

    """ the following code is needed for ray """
    def mask(self):
        return self.env.mask()
    
    def score(self, *args):
        return self.env.score()

    def epslen(self, *args):
        return self.env.epslen()

    def already_done(self):
        return self.env.already_done()

    def game_over(self):
        return self.env.game_over()

    def get_screen(self, size=None):
        if 'atari' in self.name:
            img = self.env.get_screen()
        else:
            img = self.env.render(mode='rgb_array')
        
        if size is not None:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            
        return img


class EnvVec(EnvVecBase):
    def __init__(self, config, env_fn=make_env):
        self.n_envs = n_envs = config['n_envs']
        self.name = config['name']
        self.envs = [env_fn(config) for i in range(n_envs)]
        self.env = self.envs[0]
        if 'seed' in config:
            [env.seed(config['seed'] + i) 
                for i, env in enumerate(self.envs)
                if hasattr(env, 'seed')]
        self.max_episode_steps = self.env.max_episode_steps

    def random_action(self, *args, **kwargs):
        return np.array([env.action_space.sample() for env in self.envs], copy=False)

    def reset(self, idxes=None, **kwargs):
        idxes = self._get_idxes(idxes)
        if kwargs:
            for k, v in kwargs.items():
                if isscalar(v):
                    kwargs[k] = np.tile(v, self.n_envs)
            kwargs = [dict(x) for x in zip(*[itertools.product([k], v) 
                        for k, v in kwargs.items()])]
        else:
            kwargs = [dict() for _ in idxes]
        obs = [self.envs[i].reset(**kw) for i, kw in zip(idxes, kwargs)]
        
        return self._convert_batch_obs(obs)
    
    def step(self, actions, **kwargs):
        obs, reward, done, info = _envvec_step(self.envs, actions, **kwargs)

        return (self._convert_batch_obs(obs), 
                np.array(reward, dtype=np.float32), 
                np.array(done, dtype=np.float32), 
                info)

    def mask(self):
        return np.array([env.mask() for env in self.envs], dtype=np.float32)

    def score(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return [self.envs[i].score() for i in idxes]

    def epslen(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return [self.envs[i].epslen() for i in idxes]

    def already_done(self):
        return np.array([env.already_done() for env in self.envs], dtype=np.bool)

    def game_over(self):
        return np.array([env.game_over() for env in self.envs], dtype=np.bool)
        
    def get_screen(self, size=None):
        if 'atari' in self.name:
            imgs = np.array([env.get_screen() for env in self.envs], copy=False)
        else:
            imgs = np.array([env.render(mode='rgb_array') for env in self.envs],
                            copy=False)

        if size is not None:
            imgs = np.array([cv2.resize(i, size, interpolation=cv2.INTER_AREA) 
                            for i in imgs])
        
        return imgs


class EfficientEnvVec(EnvVec):
    """ Designed for efficient evaluation only """
    def reset(self, idxes=None, **kwargs):
        # reset all envs and omit kwargs intentionally
        assert idxes is None, idxes
        self.env_ids = np.arange(self.n_envs)
        return super().reset()

    def random_action(self, *args, **kwargs):
        return [self.envs[i].action_space.sample() for i in self.env_ids]
        
    def step(self, actions, **kwargs):
        # intend to omit kwargs as EfficientEnvVec is only used in evaluation
        envs = [self.envs[i] for i in self.env_ids]
        obs, reward, done, info = _envvec_step(envs, actions)
            
        out = list(zip(*[(id_, o, r, d, i)
                for e, id_, o, r, d, i in zip(envs, self.env_ids, obs, reward, done, info) 
                if not e.game_over()]))
        if out:
            self.env_ids, obs, reward, done, info = out
            for id_, i in zip(self.env_ids, info):
                i['env_id'] = id_

            return (self._convert_batch_obs(obs), 
                    np.array(reward, dtype=np.float32), 
                    np.array(done, dtype=np.float32), 
                    info)
        else:
            return None, 0, True, {}


class RayEnvVec(EnvVecBase):
    def __init__(self, EnvType, config, env_fn=make_env):
        self.name = config['name']
        self.n_workers= config['n_workers']
        self.envsperworker = config['n_envs']
        self.n_envs = self.envsperworker * self.n_workers
        RayEnvType = ray.remote(EnvType)
        # leave the name "envs" for consistency, albeit workers seems more appropriate
        if 'seed' in config:
            self.envs = [config.update({'seed': 100*i}) or RayEnvType.remote(config, env_fn) 
                    for i in range(self.n_workers)]
        else:
            self.envs = [RayEnvType.remote(config, env_fn) 
                    for i in range(self.n_workers)]

        self.env = EnvType(config, env_fn)
        self.max_episode_steps = self.env.max_episode_steps

    def reset(self, idxes=None):
        obs = self._remote_call('reset', idxes)
        return self._convert_batch_obs(obs)

    def random_action(self, *args, **kwargs):
        return np.reshape(ray.get([env.random_action.remote() for env in self.envs]), 
                          (self.n_envs, *self.action_shape))

    def step(self, actions, **kwargs):
        actions = np.squeeze(actions.reshape(self.n_workers, self.envsperworker, *self.action_shape))
        if kwargs:
            kwargs = dict([(k, np.squeeze(v.reshape(self.n_workers, self.envsperworker, -1))) for k, v in kwargs.items()])
            kwargs = [dict(x) for x in zip(*[itertools.product([k], v) for k, v in kwargs.items()])]
        else:
            kwargs = [dict() for _ in range(self.n_workers)]

        obs, reward, done, info = zip(*ray.get([env.step.remote(a, **kw) for env, a, kw in zip(self.envs, actions, kwargs)]))
        obs = tf.nest.flatten(obs)
        if self.envsperworker > 1:
            info = itertools.chain(*info)
        return (self._convert_batch_obs(obs),
                np.reshape(reward, self.n_envs).astype(np.float32), 
                np.reshape(done, self.n_envs).astype(np.float32),
                info)

    def mask(self):
        """ Get mask at the current step. Should only be called after self.step """
        return np.reshape(ray.get([env.mask.remote() for env in self.envs]), self.n_envs)

    def score(self, idxes=None):
        return self._remote_call('score', idxes, return_numpy=True)

    def epslen(self, idxes=None):
        return self._remote_call('epslen', idxes, return_numpy=True)
        
    def already_done(self, idxes=None):
        return self._remote_call('already_done', idxes, return_numpy=True)

    def game_over(self, idxes=None):
        return self._remote_call('game_over', idxes, return_numpy=True)

    def _remote_call(self, name, idxes, return_numpy=False):
        method = lambda e: getattr(e, name)
        if idxes is None:
            val = tf.nest.flatten(ray.get([method(e).remote() for e in self.envs]), self.n_envs)
        else:
            new_idxes = [[] for _ in range(self.n_workers)]
            [new_idxes[i // self.envsperworker].append(i % self.envsperworker) for i in idxes]
            val = tf.nest.flatten(ray.get([method(self.envs[i]).remote(j) for i, j in enumerate(new_idxes)]))
        
        if return_numpy:
            return np.array(val)
        else:
            return val

    def close(self):
        del self


def _envvec_step(envvec, actions, **kwargs):
    if kwargs:
        for k, v in kwargs.items():
            if isscalar(v):
                kwargs[k] = np.tile(v, actions.shape[0])
        kwargs = [dict(x) for x in zip(*[itertools.product([k], v) for k, v in kwargs.items()])]
        return zip(*[env.step(a, **kw) for env, a, kw in zip(envvec, actions, kwargs)])
    else:
        return zip(*[env.step(a) for env, a in zip(envvec, actions)])


if __name__ == '__main__':
    # performance test
    config = dict(
        name='atari_breakout',
        seed=0,
        life_done=False,
    )
    from utility.run import run
    import matplotlib.pyplot as plt
    env = create_env(config)
    obs = env.reset()
    def fn(env, step, **kwargs):
        assert step % env.frame_skip == 0
        if env.already_done():
            print(step, env.already_done(), env.lives, env.game_over())
    run(env, env.random_action, 0, fn=fn, nsteps=10000)
    