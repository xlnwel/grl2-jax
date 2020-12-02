import itertools
import numpy as np
import cv2
import gym
import ray

from env.wrappers import *
from env import atari
from env import procgen
from env import dmc


def make_env(config):
    # config = config.copy()    # do not copy to make changes visible from the outside
    env_name = config['name'].lower()
    
    if env_name.startswith('atari'):
        env = atari.make_atari_env(config)
    elif env_name.startswith('procgen'):
        env = procgen.make_procgen_env(config)
    elif env_name.startswith('dmc'):
        env = dmc.make_dmc_env(config)
    else:
        env = gym.make(config['name']).env
        env = Dummy(env)
        config.setdefault('max_episode_steps', env.spec.max_episode_steps)
        if config.get('frame_skip'):
            env = FrameSkip(env, config['frame_skip'])
    env = post_wrap(env, config)
    if config.get('reward_scale') or config.get('reward_clip'):
        env = RewardHack(env, **config)

    return env


class Env(gym.Wrapper):
    def __init__(self, config, env_fn=make_env):
        self.env = env_fn(config)
        if 'seed' in config and hasattr(self.env, 'seed'):
            self.env.seed(config['seed'])
        self.name = config['name']
        self.max_episode_steps = self.env.max_episode_steps
        self.n_envs = 1
        self.env_type = 'Env'
        super().__init__(self.env)

    def reset(self, idxes=None):
        return self.env.reset()

    def random_action(self, *args, **kwargs):
        action = self.env.action_space.sample()
        return action
        
    def step(self, action, **kwargs):
        output = self.env.step(action, **kwargs)
        return output

    """ the following code is needed for ray """
    def score(self, *args):
        return self.env.score()

    def epslen(self, *args):
        return self.env.epslen()

    def mask(self, *args):
        return self.env.mask()

    def prev_obs(self):
        return self.env.prev_obs()

    def info(self):
        return self.env.info()

    def game_over(self):
        return self.env.game_over()

    def get_screen(self, size=None):
        if hasattr(self.env, 'get_screen'):
            img = self.env.get_screen()
        else:
            img = self.env.render(mode='rgb_array')

        if size is not None and size != img.shape[:2]:
            # cv2 receive size of form (width, height)
            img = cv2.resize(img, size[::-1], interpolation=cv2.INTER_AREA)
            
        return img


class EnvVecBase(gym.Wrapper):
    def __init__(self):
        self.env_type = 'EnvVec'
        super().__init__(self.env)

    def _convert_batch_obs(self, obs):
        if obs != []:
            if isinstance(obs[0], np.ndarray):
                obs = np.reshape(obs, [-1, *self.obs_shape])
            else:
                obs = list(obs)
        return obs

    def _get_idxes(self, idxes):
        if idxes is None:
            idxes = list(range(self.n_envs))
        elif isinstance(idxes, int):
            idxes = [idxes]
        return idxes


class EnvVec(EnvVecBase):
    def __init__(self, config, env_fn=make_env):
        self.n_envs = n_envs = config.pop('n_envs', 1)
        self.name = config['name']
        self.envs = [env_fn(config) for i in range(n_envs)]
        self.env = self.envs[0]
        if 'seed' in config:
            [env.seed(config['seed'] + i) 
                for i, env in enumerate(self.envs)
                if hasattr(env, 'seed')]
        self.max_episode_steps = self.env.max_episode_steps
        super().__init__()

    def random_action(self, *args, **kwargs):
        return np.array([env.action_space.sample() for env in self.envs], copy=False)

    def reset(self, idxes=None, **kwargs):
        idxes = self._get_idxes(idxes)
        obs, reward, done, reset = zip(*[self.envs[i].reset() for i in idxes])

        return EnvOutput(
            self._convert_batch_obs(obs),
            np.array(reward), 
            np.array(done),
            np.array(reset))

    def step(self, actions, **kwargs):
        return self._envvec_op('step', action=actions, **kwargs)

    def score(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return [self.envs[i].score() for i in idxes]

    def epslen(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return [self.envs[i].epslen() for i in idxes]
    
    def mask(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return np.array([self.envs[i].mask() for i in idxes])

    def game_over(self):
        return np.array([env.game_over() for env in self.envs], dtype=np.bool)

    def prev_obs(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return [self.envs[i].prev_obs() for i in idxes]

    def info(self, idxes=None):
        idxes = self._get_idxes(idxes)
        return [self.envs[i].info() for i in idxes]

    def output(self, idxes=None):
        idxes = self._get_idxes(idxes)
        obs, reward, done, reset = zip(*[self.envs[i].output() for i in idxes])

        return EnvOutput(
            self._convert_batch_obs(obs),
            np.array(reward), 
            np.array(done),
            np.array(reset))

    def get_screen(self, size=None):
        if hasattr(self.env, 'get_screen'):
            imgs = np.array([env.get_screen() for env in self.envs], copy=False)
        else:
            imgs = np.array([env.render(mode='rgb_array') for env in self.envs],
                            copy=False)

        if size is not None:
            # cv2 receive size of form (width, height)
            imgs = np.array([cv2.resize(i, size[::-1], interpolation=cv2.INTER_AREA) 
                            for i in imgs])
        
        return imgs

    def _envvec_op(self, name, **kwargs):
        method = lambda e: getattr(e, name)
        if kwargs:
            kwargs = [dict(x) for x in zip(*[itertools.product([k], v) 
                for k, v in kwargs.items()])]
            obs, reward, done, reset = \
                zip(*[method(env)(**kw) for env, kw in zip(self.envs, kwargs)])
        else:
            obs, reward, done, reset = \
                zip(*[method(env)() for env in self.envs])

        return EnvOutput(
            self._convert_batch_obs(obs),
            np.array(reward), 
            np.array(done),
            np.array(reset))


class RayEnvVec(EnvVecBase):
    def __init__(self, EnvType, config, env_fn=make_env):
        self.name = config['name']
        config['np_obs'] = True
        self.n_workers= config.get('n_workers', 1)
        self.envsperworker = config.get('n_envs', 1)
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
        super().__init__()

    def reset(self, idxes=None):
        output = self._remote_call('reset', idxes, single_output=False)

        if isinstance(self.env, Env):
            output = [np.stack(o, 0) for o in output]
        else:
            output = [np.concatenate(o, 0) for o in output]
        return EnvOutput(*output)

    def random_action(self, *args, **kwargs):
        return np.reshape(ray.get([env.random_action.remote() for env in self.envs]), 
                          (self.n_envs, *self.action_shape))

    def step(self, actions, **kwargs):
        actions = np.squeeze(actions.reshape(self.n_workers, self.envsperworker, *self.action_shape))
        if kwargs:
            kwargs = dict([(k, np.squeeze(v.reshape(self.n_workers, self.envsperworker, -1))) for k, v in kwargs.items()])
            kwargs = [dict(x) for x in zip(*[itertools.product([k], v) for k, v in kwargs.items()])]
            output = ray.get([env.step.remote(a, **kw) 
                for env, a, kw in zip(self.envs, actions, kwargs)])
        else:
            output = ray.get([env.step.remote(a) 
                for env, a in zip(self.envs, actions)])
        output = list(zip(*output))

        if isinstance(self.env, Env):
            output = [np.stack(o, 0) for o in output]
        else:
            output = [np.concatenate(o, 0) for o in output]
        return EnvOutput(*output)

    def score(self, idxes=None):
        return self._remote_call('score', idxes)

    def epslen(self, idxes=None):
        return self._remote_call('epslen', idxes)
        
    def game_over(self, idxes=None):
        return self._remote_call('game_over', idxes)

    def prev_obs(self, idxes=None):
        return self._remote_call('prev_obs', idxes)

    def info(self, idxes=None):
        return self._remote_call('info', idxes)
    
    def output(self, idxes=None):
        output = self._remote_call('output', idxes, single_output=False)

        if isinstance(self.env, Env):
            output = [np.stack(o, 0) for o in output]
        else:
            output = [np.concatenate(o, 0) for o in output]
        return EnvOutput(*output)

    def _remote_call(self, name, idxes, single_output=True):
        """
        single_output: if the call produces only one output
        """
        method = lambda e: getattr(e, name)
        if idxes is None:
            output = ray.get([method(e).remote() for e in self.envs])
        else:
            if isinstance(self.env, Env):
                output = ray.get([method(self.envs[i]).remote() for i in idxes])
            else:
                new_idxes = [[] for _ in range(self.n_workers)]
                for i in idxes:
                    new_idxes[i // self.envsperworker].append(i % self.envsperworker)
                output = ray.get([method(self.envs[i]).remote(j) 
                    for i, j in enumerate(new_idxes) if j])
        if single_output:
            if isinstance(self.env, Env):
                return output
            # for these outputs, we expect them to be of form [[output*], [output*]]
            # and we chain them into [output*]
            return list(itertools.chain(*output))
        else:
            return list(zip(*output))

    def close(self):
        del self
