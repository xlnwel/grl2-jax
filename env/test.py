import time
import numpy as np
import ray

from env.func import create_env


if __name__ == '__main__':
    config = dict(
        name='mpe_simple_reference',
        # n_envs=10,
        max_episode_steps=100,
        n_agents=3,
        num_good_agents=3,
        num_adversaries=2,
        num_landmarks=4
    )
    # def make_env(config):
    #     env = procgen.make_procgen_env(config)
    #     return env

    if config.get('n_workers', 1) > 1:
        ray.init()
    env = create_env(config)
    print('max episode length', env.max_episode_steps)
    print('Env', env)

    def run(env):
        st = time.time()
        for _ in range(100):
            a = env.random_action()
            _, r, d, _ = env.step(a)
            env.render()
            print('reward', r)
            time.sleep(.1)
            if not np.any(d):
                env.reset()
        return time.time() - st

    print("Ray env:", run(env))

    if ray.is_initialized():
        ray.shutdown()
