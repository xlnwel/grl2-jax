import numpy as np
import ray

from core.tf_config import configure_gpu, configure_precision, silence_tf_logs
from utility.utils import Every
from utility.signal import sigint_shutdown_ray
from utility.timer import Timer
from env.gym_env import create_env
from algo.ppo.eval import evaluate

def import_model_fn(algorithm):
    if algorithm == 'ppo':
        from algo.ppo.nn import create_model
    elif algorithm == 'ppo1':
        from algo.ppo1.nn import create_model
    elif algorithm == 'ppo2':
        from algo.ppo2.nn import create_model
    else:
        raise NotImplementedError(algorithm)
    return create_model

def import_agent(algorithm):
    if algorithm == 'ppo':
        from algo.ppo.agent import Agent
    elif algorithm == 'ppo1':
        from algo.ppo1.agent import Agent
    elif algorithm == 'ppo2':
        from algo.ppo2.agent import Agent
    else:
        raise NotImplementedError(algorithm)
    return Agent

def import_run(algorithm):
    if algorithm == 'ppo':
        from algo.ppo.run import run
    elif algorithm == 'ppo1':
        from algo.ppo1.run import run
    elif algorithm == 'ppo2':
        from algo.ppo2.run import run
    else:
        raise NotImplementedError(algorithm)
    return run

def import_buffer(algorithm):
    if algorithm == 'ppo':
        from algo.ppo.buffer import PPOBuffer
    elif algorithm == 'ppo1':
        from algo.ppo1.buffer import PPOBuffer
    elif algorithm == 'ppo2':
        from algo.ppo2.buffer import PPOBuffer
    else:
        raise NotImplementedError(algorithm)
    return PPOBuffer


def train(agent, buffer, env, run):
    step = agent.global_steps.numpy()
    
    eval_env = create_env(dict(
        name=env.name, 
        video_path='video',
        log_video=False,
        n_workers=1,
        n_envs=env.n_envs,
        effective_envvec=True,
        seed=0,
    ))
    
    should_log = Every(agent.LOG_INTERVAL)
    obs = env.reset()
    while step < agent.MAX_STEPS:
        agent.set_summary_step(step)
        step, obs = run(env, agent, buffer, step, obs)

        if should_log(step):
            if agent._algorithm == 'ppo2':
                state = agent.curr_state
            scores, epslens = evaluate(eval_env, agent)
            agent.store(eval_score=scores, eval_epslen=np.mean(epslens))

            agent.store(**agent.get_value('score', mean=True, std=True, min=True, max=True))
            agent.store(**agent.get_value('epslen', mean=True, std=True, min=True, max=True))
            agent.store(**agent.get_value('eval_score', mean=True, std=True, min=True, max=True))
            agent.store(**agent.get_value('eval_epslen', mean=True, std=True, min=True, max=True))

            agent.log(step)
            agent.save(steps=step)
            
            if agent._algorithm == 'ppo2':
                agent.curr_state = state

def main(env_config, model_config, agent_config, buffer_config, restore=False, render=False):
    algo = agent_config['algorithm']
    create_model = import_model_fn(algo)
    Agent = import_agent(algo)
    run = import_run(algo)
    PPOBuffer = import_buffer(algo)

    silence_tf_logs()
    configure_gpu()
    configure_precision(agent_config['precision'])

    use_ray = env_config.get('n_workers', 0) > 1
    if use_ray:
        ray.init()
        sigint_shutdown_ray()

    env = create_env(env_config, force_envvec=True)
    
    if agent_config['algorithm'] == 'ppo1':
        buffer_config['n_steps'] = env.max_episode_steps
    buffer = PPOBuffer(buffer_config)

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
