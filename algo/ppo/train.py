import ray

from utility.utils import set_global_seed
from utility.tf_utils import configure_gpu
from utility.signal import sigint_shutdown_ray
from env.gym_env import create_gym_env
from buffer.ppo_buffer import PPOBuffer
from algo.ppo.nn import PPOAC
from algo.ppo.agent import Agent


def main(env_config, agent_config, buffer_config, render=False):
    set_global_seed()
    configure_gpu()

    use_ray = env_config.get('n_workers', 0) > 1
    if use_ray:
        ray.init()
        sigint_shutdown_ray()

    # construct environment
    env = create_gym_env(env_config)
    
    n_minibatches = agent_config['n_minibatches']
    # construct model
    ac = PPOAC(env.state_shape, 
                env.action_dim, 
                env.is_action_discrete,
                env.n_envs,
                'ac')

    # construct buffer
    buffer = PPOBuffer(env.n_envs,
                       env.max_episode_steps,
                       n_minibatches,
                       env.state_shape,
                       env.state_dtype,
                       env.action_shape,
                       env.action_dtype,
                       **buffer_config)

    # construct agent
    agent = Agent('ppo', 
                agent_config, 
                env=env, 
                buffer=buffer, 
                models=[ac])

    # agent.restore()
    agent.train()

    if use_ray:
        ray.shutdown()
