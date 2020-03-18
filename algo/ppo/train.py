import numpy as np
import ray

from utility.utils import set_global_seed
from core.tf_config import configure_gpu
from utility.signal import sigint_shutdown_ray
from utility.timer import Timer
from env.gym_env import create_gym_env
from algo.ppo.buffer import PPOBuffer
from algo.ppo.run import run_trajectories
from algo.ppo.eval import evaluate

def import_model_fn(algorithm):
    if algorithm == 'ppo':
        from algo.ppo.nn import create_model
    elif algorithm == 'ppo2':
        from algo.ppo2.nn import create_model
    else:
        raise NotImplementedError(algorithm)
    return create_model

def import_agent(algorithm):
    if algorithm == 'ppo':
        from algo.ppo.agent import Agent
    elif algorithm == 'ppo2':
        from algo.ppo2.agent import Agent
    else:
        raise NotImplementedError(algorithm)
    return Agent

def train(agent, env, buffer):
    start_epoch = agent.global_steps.numpy()+1
    for epoch in range(start_epoch, agent.N_EPOCHS+1):
        agent.set_summary_step(epoch)
        with Timer(f'{agent.name} run', agent.LOG_INTERVAL):
            scores, epslens = run_trajectories(env, agent, buffer)
        agent.store(score=scores, epslen=epslens)

        with Timer(f'{agent.name} training', agent.LOG_INTERVAL):
            agent.learn_log(buffer, epoch=epoch)

        if epoch % agent.LOG_INTERVAL == 0:
            with Timer(f'{agent.name} evaluation'):
                scores, epslens = evaluate(env, agent)
            agent.store(eval_score=scores, eval_epslen=np.mean(epslens))

            agent.store(**agent.get_value('score', mean=True, std=True, min=True, max=True))
            agent.store(**agent.get_value('epslen', mean=True, std=True, min=True, max=True))
            agent.store(**agent.get_value('eval_score', mean=True, std=True, min=True, max=True))
            agent.store(**agent.get_value('eval_epslen', mean=True, std=True, min=True, max=True))

            with Timer(f'{agent.name} logging'):
                agent.log(epoch)
            with Timer(f'{agent.name} save'):
                agent.save(steps=epoch)

def main(env_config, model_config, agent_config, buffer_config, restore=False, render=False):
    algo = agent_config['algorithm']
    create_model = import_model_fn(algo)
    Agent = import_agent(algo)

    set_global_seed()
    configure_gpu()

    use_ray = env_config.get('n_workers', 0) > 1
    if use_ray:
        ray.init()
        sigint_shutdown_ray()

    env = create_gym_env(env_config)
    
    buffer = PPOBuffer(buffer_config, env.n_envs, env.max_episode_steps)

    models = create_model(
        model_config, 
        action_dim=env.action_dim, 
        is_action_discrete=env.is_action_discrete,
        n_envs=env.n_envs
    )
    
    agent = Agent(name='ppo', 
                config=agent_config, 
                models=models, 
                env=env)

    agent.save_config(dict(
        env=env_config,
        model=model_config,
        agent=agent_config,
        buffer=buffer_config
    ))

    if restore:
        agent.restore()

    train(agent, env, buffer)

    if use_ray:
        ray.shutdown()
