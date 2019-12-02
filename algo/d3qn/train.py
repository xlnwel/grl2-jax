from collections import deque
import numpy as np
import ray

from utility.utils import set_global_seed
from core.tf_config import configure_gpu
from utility.signal import sigint_shutdown_ray
from utility.timer import Timer
from utility.utils import step_str
from env.gym_env import create_gym_env
from replay.func import create_replay
from replay.data_pipline import Dataset
from algo import run
from algo.d3qn.agent import Agent
from algo.d3qn.nn import create_model


def evaluate(agent, global_steps, env):
    step = 0
    scores = []
    epslens = []
    with Timer(f'{agent.model_name} evaluation'):
        while step < agent.eval_steps:
            score, epslen = run.run_trajectory(env, agent.q1, evaluation=True)
            scores.append(score)
            epslens.append(epslen)
            step += epslen

    stats = dict(
        model_name=f'{agent.model_name}',
        timing='Eval',
        steps=step_str(global_steps), 
        score=np.mean(scores),
        score_std=np.std(scores),
        score_max=np.amax(scores),
        epslen=np.mean(epslens),
        epslen_std=np.std(epslens),
    )
    agent.log_stats(stats)

def train(agent, env, replay):
    def collect_and_learn(state, action, reward, done, next_state, step, **kwargs):
        replay.add(state=state, action=action, reward=reward, done=done, next_state=next_state)
        if step % agent.update_freq == 0:
            with Timer('learn', 10000):
                agent.learn_log(step // agent.update_freq)
    
    global_steps = agent.global_steps.numpy()
    step = 0
    episode = 0
    scores = []
    epslens = []
    with Timer(f'{agent.model_name} training'):
        while step < agent.train_steps:
            episode += 1
            agent.q1.reset_noisy()
            score, epslen = run.run_trajectory(env, agent.q1, fn=collect_and_learn, step=step)
            scores.append(score)
            epslens.append(epslen)
            step += epslen
            
            if episode % 5 == 0:        
                agent.store(
                    score=np.mean(scores),
                    score_std=np.std(scores),
                    score_max=np.amax(scores),
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
    eval_env = create_gym_env(eval_env_config)
    # construct replay
    replay_keys = ['state', 'action', 'reward', 'done', 'steps']
    replay = create_replay(replay_config, *replay_keys, state_shape=env.state_shape)
    dataset = Dataset(replay, env.state_shape, env.state_dtype, env.action_shape, env.action_dtype)

    # construct models
    models = create_model(model_config, env.state_shape, env.action_dim)

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
        collect_fn = lambda state, action, reward, done: replay.add(state, action, reward, done)
        while not replay.good_to_learn():
            run.run_trajectory(env, agent.actor, collect_fn)
    else:
        run.random_sampling(env, replay)

    print(f'Start training...')
    global_steps = agent.global_steps.numpy()
    max_steps = int(agent.max_steps)
    while global_steps < max_steps:
        global_steps = train(agent, env, replay)
        evaluate(agent, global_steps, eval_env)

