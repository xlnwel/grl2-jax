import numpy as np
import gym

from utility import pkg


class Unity:
    def __init__(self, config):
        self.env = self._create_env(config)

        self.action_shape = [a.shape for a in self.env.action_space]
        self.is_action_discrete = [isinstance(a, gym.spaces.Discrete) for a in self.env.action_space]
        self.action_dim = [a.n if c else a.shape[0] 
            for c, a in zip(self.is_action_discrete, self.env.action_space)]
        self.action_dtype = [np.int32 if c else a.dtype 
            for c, a in zip(self.is_action_discrete, self.env.action_space)]
        self.is_multi_agent = True

    def _create_env(self, config):
        UnityEnv = pkg.import_module(config['env_name'], 'env.unity_env').UnityEnv
        return UnityEnv(**config)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self.env, name)


if __name__ == '__main__':
    config = dict(
        env_name='dummy2',
        uid2aid=[0, 1, 1, 1, 1],
        max_episode_steps=100,
        n_envs=4,
    )
    from utility.display import print_dict, print_dict_info
    def print_dict_info(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f'{prefix} {k}')
                print_dict_info(v, prefix+'\t')
            elif isinstance(v, tuple):
                # namedtuple is assumed
                print(f'{prefix} {k}')
                print_dict_info(v._asdict(), prefix+'\t')
            else:
                print(f'{prefix} {k}: {v.shape} {v.dtype}')

    env = Unity(config)
    observations = env.reset()
    print('reset observations')
    for i, o in enumerate(observations):
        print_dict_info(o, f'\tagent{i}')
    for k in range(1, 3):
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
