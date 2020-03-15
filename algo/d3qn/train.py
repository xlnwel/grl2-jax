from collections import deque
import numpy as np
import ray

from utility.utils import set_global_seed
from core.tf_config import configure_gpu
from utility.signal import sigint_shutdown_ray
from utility.timer import TBTimer
from utility.utils import step_str
from env.gym_env import create_gym_env
from replay.func import create_replay
from replay.data_pipline import Dataset
from algo.run import run, random_sampling
from algo.d3qn.agent import Agent
from algo.d3qn.nn import create_model


def evaluate(agent, global_steps, env):
    step = 0
    scores = []
    epslens = []
    with TBTimer(f'evaluation', agent.LOG_INTERVAL, to_log=agent.TIMER):
        while step < agent.eval_steps:
            score, epslen = run(env, agent.q1, evaluation=True, timer=agent.TIMER)
            scores.append(score)
            epslens.append(epslen)
            step += np.sum(epslen)

    stats = dict(
        model_name=f'{agent.model_name}',
        timing='Eval',
        steps=global_steps, 
        score=np.mean(scores),
        score_std=np.std(scores),
        score_max=np.max(scores),
        epslen=np.mean(epslens),
        epslen_std=np.std(epslens),
    )
    agent.log_stats(stats)

def train(agent, env, replay):
    def collect_and_learn(obs, action, reward, done, next_obs, step, **kwargs):
        replay.add(obs=obs, action=action, reward=reward, done=done, next_obs=next_obs)
        if step % agent.update_freq == 0:
            with TBTimer('learn', agent.LOG_INTERVAL, to_log=agent.TIMER):
                agent.learn_log(step // agent.update_freq)
    
    global_steps = agent.global_steps.numpy()
    step = 0
    episode = 0
    scores = []
    epslens = []
    with TBTimer(f'training', agent.LOG_INTERVAL, to_log=agent.TIMER):
        while step < agent.train_steps:
            episode += 1
            agent.q1.reset_noisy()
            score, epslen = run(env, agent.q1, fn=collect_and_learn, step=step, timer=agent.TIMER)
            scores.append(score)
            epslens.append(epslen)
            step += epslen
            
            if episode % 5 == 0:        
                agent.store(
                    score=np.mean(scores),
                    score_std=np.std(scores),
                    score_max=np.max(scores),
                    epslen=np.mean(epslens),
                    epslen_std=np.std(epslens),
                )
                cur_step = global_steps + step
                agent.log(cur_step, 'Train')
                agent.global_steps.assign(cur_step)
                agent.save()

    return agent.global_steps.numpy()
    

def main(env_config, model_config, agent_config, replay_config, restore=False, render=False):
    set_global_seed()
    configure_gpu()

    env = create_gym_env(env_config)
    eval_env_config = env_config.copy()
    # eval_env_config['log_video'] = True
    eval_env_config['seed'] = np.random.randint(100, 1000)
    eval_env_config['n_envs'] = 50
    eval_env_config['efficient_envvec'] = True
    eval_env = create_gym_env(eval_env_config)
    # construct replay
    replay_keys = ['obs', 'action', 'reward', 'done', 'steps']
    replay = create_replay(replay_config, *replay_keys, obs_shape=env.obs_shape)
    dataset = Dataset(replay, env.obs_shape, env.obs_dtype, env.action_shape, env.action_dtype, env.action_dim)

    # construct models
    models = create_model(model_config, env.obs_shape, env.action_dim)

    # construct agent
    agent = Agent(name='q', 
                config=agent_config, 
                models=models, 
                dataset=dataset, 
                env=env)
    
    agent.save_config(dict(
        env=env_config,
        model=model_config,
        agent=agent_config,
        replay=replay_config
    ))

    if restore:
        agent.restore()
        collect_fn = lambda obs, action, reward, done: replay.add(obs, action, reward, done)
        while not replay.good_to_learn():
            run_trajectory(env, agent.actor, collect_fn)
    else:
        random_sampling(env, replay)

    print(f'Start training...')
    global_steps = agent.global_steps.numpy()
    MAX_STEPS = int(agent.MAX_STEPS)
    while global_steps < MAX_STEPS:
        global_steps = train(agent, env, replay)
        evaluate(agent, global_steps, eval_env)

