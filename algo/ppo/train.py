import time
import numpy as np
import ray

from core.tf_config import configure_gpu, configure_precision, silence_tf_logs
from utility.utils import Every
from utility.ray_setup import sigint_shutdown_ray
from utility.graph import video_summary
from utility.run import evaluate
from utility.timer import Timer
from run import pkg
from env.gym_env import create_env



def train(agent, env, eval_env, buffer, run):
    print(env, eval_env)
    step = agent.global_steps.numpy()
    
    to_log = Every(agent.LOG_INTERVAL)
    obs = env.reset()
    step = agent.global_steps.numpy()
    start_env_step = step
    train_step = 0
    start_train_step = 0
    start_time = time.time()
    while step < agent.MAX_STEPS:
        agent.set_summary_step(step)
        step, obs = run(env, agent, buffer, step, obs)
        train_step += agent.N_UPDATES * agent.N_MBS
        if to_log(step):
            if agent._algorithm >= 'ppo2':
                state = agent.state
            duration = time.time()-start_time
            agent.store(
                fps=(step-start_env_step)/duration,
                tps=(train_step-start_train_step)/duration,
            )
            scores, epslens, video = evaluate(eval_env, agent)
            # video_summary(f'{agent.name}/sim', video, step)
            agent.store(eval_score=scores, eval_epslen=np.mean(epslens))

            agent.log(step)
            agent.save(steps=step)
            
            if agent._algorithm >= 'ppo2':
                agent.state = state

            start_train_step = train_step
            start_env_step = step
            start_time = time.time()

def main(env_config, model_config, agent_config, buffer_config):
    algo = agent_config['algorithm']
    create_model, Agent = pkg.import_agent(agent_config)
    run = pkg.import_module('run', algo=algo).run
    PPOBuffer = pkg.import_module('buffer', algo=algo).PPOBuffer

    silence_tf_logs()
    configure_gpu()
    configure_precision(agent_config['precision'])

    use_ray = env_config.get('n_workers', 1) > 1
    if use_ray:
        ray.init()
        sigint_shutdown_ray()

    env = create_env(env_config)
    eval_env_config = env_config.copy()
    eval_env_config['seed'] += 100
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

    train(agent, env, eval_env, buffer, run)

    if use_ray:
        ray.shutdown()
