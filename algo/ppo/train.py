import numpy as np
import ray

from core.tf_config import configure_gpu, configure_precision, silence_tf_logs
from utility.utils import Every
from utility.ray_setup import sigint_shutdown_ray
from utility.timer import Timer
from env.gym_env import create_env
from algo.ppo.eval import evaluate, import_model_fn, import_agent


def import_run(algorithm):
    if algorithm == 'ppo':
        from algo.ppo.run import run
    elif algorithm == 'ppo1':
        from algo.ppo1.run import run
    elif algorithm == 'ppo2':
        from algo.ppo2.run import run
    elif algorithm == 'ppo3':
        from algo.ppo3.run import run
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
    elif algorithm == 'ppo3':
        from algo.ppo3.buffer import PPOBuffer
    else:
        raise NotImplementedError(algorithm)
    return PPOBuffer


def train(agent, env, eval_env, buffer, run):
    step = agent.global_steps.numpy()
    
    to_log = Every(agent.LOG_INTERVAL)
    obs = env.reset()
    while step < agent.MAX_STEPS:
        agent.set_summary_step(step)
        step, obs = run(env, agent, buffer, step, obs)

        if to_log(step):
            if agent._algorithm >= 'ppo2':
                state = agent.state
                
            scores, epslens = evaluate(eval_env, agent)
            agent.store(eval_score=scores, eval_epslen=np.mean(epslens))

            agent.log(step)
            agent.save(steps=step)
            
            if agent._algorithm >= 'ppo2':
                agent.state = state

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
    eval_env_config = env_config.copy()
    eval_env_config['n_workers'] = 1
    eval_env_config['n_envs'] = 1
    eval_env = create_env(eval_env_config, force_envvec=True)

    buffer_config['n_envs'] = env_config['n_envs']
    if algo == 'ppo1':
        buffer_config['n_steps'] = env.max_episode_steps
    buffer = PPOBuffer(buffer_config)

    models = create_model(
        model_config, 
        action_dim=env.action_dim, 
        is_action_discrete=env.is_action_discrete,
    )
    
    agent_config['N_MBS'] = buffer_config['n_mbs']
    agent_config['N_STEPS'] = buffer_config['n_steps']
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

    train(agent, env, eval_env, buffer, run)

    if use_ray:
        ray.shutdown()
