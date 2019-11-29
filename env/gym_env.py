""" Implementation of single process environment """
import numpy as np
import gym
import ray

from utility import tf_distributions
from utility.display import pwc, assert_colorize
from utility.utils import to_int
from utility.timer import Timer
from env.wrappers import TimeLimit, EnvStats

def action_dist_type(env):
    if isinstance(env.action_space, gym.spaces.Discrete):
        return tf_distributions.Categorical
    elif isinstance(env.action_space, gym.spaces.Box):
        return tf_distributions.DiagGaussian
    else:
        raise NotImplementedError


class EnvBase:
    @property
    def is_action_discrete(self):
        return self.env.is_action_discrete

    @property
    def state_shape(self):
        return self.env.state_shape

    @property
    def state_dtype(self):
        return self.env.state_dtype

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def action_shape(self):
        return self.env.action_shape

    @property
    def action_dtype(self):
        return self.env.action_dtype

    @property
    def action_dim(self):
        return self.env.action_dim


class GymEnv(EnvBase):
    def __init__(self, config):
        self.name = config['name']
        env = gym.make(self.name)
        env.seed(config.get('seed', 42))
        self.max_episode_steps = config.get('max_episode_steps', env.spec.max_episode_steps)
        if self.max_episode_steps != env.spec.max_episode_steps:
            env = TimeLimit(env, self.max_episode_steps)
        if 'log_video' in config and config['log_video']:
            pwc(f'video will be logged at {config["video_path"]}', color='cyan')
            env = gym.wrappers.Monitor(env, config['video_path'], force=True)
    
        self.env = env = EnvStats(env)

    @property
    def n_envs(self):
        return 1

    def reset(self):
        return self.env.reset()

    def random_action(self):
        action = self.env.action_space.sample()
        return action
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

    def render(self):
        return self.env.render()

    def get_mask(self):
        """ Get mask at the current step. Should only be called after self.step """
        return self.env.get_mask()

    def get_score(self):
        return self.env.get_score()

    def get_epslen(self):
        return self.env.get_epslen()


class GymEnvVecBase(EnvBase):
    def __init__(self, config):
        self.n_envs = n_envs = config['n_envs']
        self.name = config['name']
        envs = [gym.make(self.name) for i in range(n_envs)]
        [env.seed(config['seed'] + i) for i, env in enumerate(envs)]
        # print(config['seed'])
        self.max_episode_steps = config.get('max_episode_steps', envs[0].spec.max_episode_steps)
        if self.max_episode_steps != envs[0].spec.max_episode_steps:
            envs = [TimeLimit(env, self.max_episode_steps) for env in envs]
        self.envs = [EnvStats(env) for env in envs]
        self.env = self.envs[0]
    
    def random_action(self):
        return np.asarray([env.action_space.sample() for env in self.envs])

    def reset(self):
        return np.asarray([env.reset() for env in self.envs])
    
    def step(self, actions):
        step_imp = lambda envs, actions: list(zip(*[env.step(a) for env, a in zip(envs, actions)]))
        
        state, reward, done, info = step_imp(self.envs, actions)
        
        return (np.asarray(state), 
                np.reshape(reward, [self.n_envs, 1]), 
                np.reshape(done, [self.n_envs, 1]), 
                info)

    def get_mask(self):
        """ Get mask at the current step. Should only be called after self.step """
        return np.reshape([env.get_mask() for env in self.envs], (self.n_envs, 1))

    def get_score(self):
        return np.asarray([env.get_score() for env in self.envs])

    def get_epslen(self):
        return np.asarray([env.get_epslen() for env in self.envs])

    def close(self):
        del self

class GymEnvVec(EnvBase):
    def __init__(self, EnvType, config):
        self.name = config['name']
        self.n_workers= config['n_workers']
        self.envsperworker = config['n_envs']
        self.n_envs = self.envsperworker * self.n_workers

        RayEnvType = ray.remote(num_cpus=1)(EnvType)
        # leave the name envs for consistency, albeit workers seems more appropriate
        self.envs = [config.update({'seed': 100*i}) or RayEnvType.remote(config.copy()) 
                    for i in range(self.n_workers)]

        self.env = GymEnv(config)
        self.max_episode_steps = self.env.max_episode_steps

    def reset(self):
        return np.reshape(ray.get([env.reset.remote() for env in self.envs]), 
                          (self.n_envs, *self.state_shape))

    def random_action(self):
        return np.reshape(ray.get([env.random_action.remote() for env in self.envs]), 
                          (self.n_envs, *self.action_shape))

    def step(self, actions):
        actions = np.reshape(actions, (self.n_workers, self.envsperworker, *self.action_shape))
        state, reward, done, info = list(zip(*ray.get([env.step.remote(a) for a, env in zip(actions, self.envs)])))

        return (np.reshape(state, (self.n_envs, *self.state_shape)), 
                np.reshape(reward, (self.n_envs, 1)), 
                np.reshape(done, (self.n_envs, 1)),
                info)

    def get_mask(self):
        """ Get mask at the current step. Should only be called after self.step """
        return np.reshape(ray.get([env.get_mask.remote() for env in self.envs]), (self.n_envs, 1))

    def get_score(self):
        return np.reshape(ray.get([env.get_score.remote() for env in self.envs]), self.n_envs)

    def get_epslen(self):
        return np.reshape(ray.get([env.get_epslen.remote() for env in self.envs]), self.n_envs)

    def close(self):
        del self


def create_gym_env(config):
    # manually use GymEnvVecBaes for easy environments
    EnvType = GymEnv if config.get('n_envs', 1) == 1 else GymEnvVecBase
    if 'n_workers' not in config or config['n_workers'] == 1:
        return EnvType(config)
    else:
        return GymEnvVec(EnvType, config)


if __name__ == '__main__':
    # performance test
    default_config = dict(
        name='BipedalWalker-v2', # Pendulum-v0, CartPole-v0
        video_path='video',
        log_video=False,
        n_workers=8,
        n_envs=2,
        seed=0
    )

    ray.init()
    config = default_config.copy()
    n = config['n_workers']
    envvec = create_gym_env(config)
    print('Env type', type(envvec))
    actions = envvec.random_action()
    with Timer(f'envvec {n} workers'):
        states = envvec.reset()
        for _ in range(envvec.max_episode_steps):
            states, rewards, dones, _ = envvec.step(actions)
    print(envvec.get_epslen())
    envvec.close()
    ray.shutdown()

    config = default_config.copy()
    config['n_envs'] *= config['n_workers']
    del config['n_workers']
    envs = create_gym_env(config)
    print('Env type', type(envs))
    actions = envs.random_action()
    with Timer('envvecbase'):
        states = envs.reset()
        for _ in range(envs.max_episode_steps):
            state, reward, done, _ = envs.step(actions)
            
            if np.all(done):
                break
            
    print(envs.get_epslen())
    
    # ray.shutdown()