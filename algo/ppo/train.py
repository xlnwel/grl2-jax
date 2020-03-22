import numpy as np
import ray

from core.tf_config import configure_gpu
from utility.utils import set_global_seed
from utility.signal import sigint_shutdown_ray
from utility.timer import Timer
from env.gym_env import create_env
from algo.ppo.buffer import PPOBuffer
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

def import_run(algorithm):
    if algorithm == 'ppo':
        from algo.ppo.run import run_trajectories
    elif algorithm == 'ppo2':
        from algo.ppo2.run import run_trajectories
    else:
        raise NotImplementedError(algorithm)
    return run_trajectories

def import_buffer(algorithm):
    if algorithm == 'ppo':
        from algo.ppo.buffer import PPOBuffer
    elif algorithm == 'ppo2':
        from algo.ppo2.buffer import PPOBuffer
    else:
        raise NotImplementedError(algorithm)
    return PPOBuffer


def train(agent, buffer, env, run):
    start_epoch = agent.global_steps.numpy()+1
    
    eval_env = create_env(dict(
        name=env.name, 
        video_path='video',
        log_video=False,
        n_workers=1,
        n_envs=env.n_envs,
        effective_envvec=True,
        seed=0,
    ))

    for epoch in range(start_epoch, agent.N_EPOCHS+1):
        agent.set_summary_step(epoch)
        with Timer(f'{agent.name} run', agent.LOG_INTERVAL):
            scores, epslens = run(env, agent, buffer, epoch=epoch)
        agent.store(score=scores, epslen=epslens)

        if agent._algorithm == 'ppo':
            with Timer(f'{agent.name} training', agent.LOG_INTERVAL):
                agent.learn_log(buffer, epoch=epoch)

        if epoch % agent.LOG_INTERVAL == 0:
            with Timer(f'{agent.name} evaluation'):
                scores, epslens = evaluate(eval_env, agent)
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
    run = import_run(algo)
    PPOBuffer = import_buffer(algo)

    set_global_seed()
    configure_gpu()

    use_ray = env_config.get('n_workers', 0) > 1
    if use_ray:
        ray.init()
        sigint_shutdown_ray()

    env = create_env(env_config)
    
    buffer = PPOBuffer(buffer_config, env.n_envs, env.max_episode_steps)

    models = create_model(
        model_config, 
        action_dim=env.action_dim, 
        is_action_discrete=env.is_action_discrete,
        n_envs=env.n_envs
    )
    
    agent = Agent(name=algo, 
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

    train(agent, buffer, env, run)

    if use_ray:
        ray.shutdown()
