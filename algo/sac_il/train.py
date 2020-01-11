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
from algo.sac.agent import Agent
from algo.sac.nn import create_model


LOG_INTERVAL = 1000

def train(agent, env, replay):
    def collect_and_learn(state, action, reward, done, next_state, **kwargs):
        replay.add(state=state, action=action, 
                reward=reward, done=done, next_state=next_state)
        agent.learn_log()

    eval_env = create_gym_env(dict(
        name=env.name, 
        video_path='video',
        log_video=False,
        n_workers=1,
        n_envs=50,
        effective_envvec=True,
        seed=np.random.randint(100, 10000),
    ))
    start_step = agent.global_steps.numpy() + 1
    scores = deque(maxlen=100)
    epslens = deque(maxlen=100)
    print('Training started...')
    step = start_step
    log_step = LOG_INTERVAL
    while step < int(agent.max_steps):
        agent.set_summary_step(step)
        with TBTimer(f'trajectory', agent.TIME_INTERVAL, to_log=agent.timer):
            score, epslen = run(env, agent.actor, fn=collect_and_learn, timer=agent.timer)
        step += epslen
        scores.append(score)
        epslens.append(epslen)
        
        if step > log_step:
            log_step += LOG_INTERVAL
            agent.save(steps=step)

            with TBTimer(f'evaluation', agent.TIME_INTERVAL, to_log=agent.timer):
                eval_scores, eval_epslens = run(eval_env, agent.actor, evaluation=True, timer=agent.timer, name='eval')
            agent.store(
                score=np.mean(eval_scores),
                score_std=np.std(eval_scores),
                score_max=np.max(eval_scores),
                epslen=np.mean(eval_epslens),
                epslen_std=np.std(eval_epslens)
            )
            agent.log(step, timing='Eval')

            stats = dict(
                model_name=f'{agent.model_name}',
                timing='Train',
                steps=step_str(step), 
                score=np.mean(scores),
                score_std=np.std(scores),
                score_max=np.max(scores),
                epslen=np.mean(epslens),
                epslen_std=np.std(epslens),
            )
            agent.log_stats(stats)

def main(env_config, model_config, agent_config, replay_config, restore=False, render=False):
    set_global_seed()
    configure_gpu()

    env = create_gym_env(env_config)

    # construct replay
    replay_keys = ['state', 'action', 'reward', 'done', 'steps']
    replay = create_replay(replay_config, *replay_keys, state_shape=env.state_shape)
    dataset = Dataset(replay, env.state_shape, env.state_dtype, env.action_shape, env.action_dtype, env.action_dim)

    # construct models
    models = create_model(
        model_config, 
        state_shape=env.state_shape, 
        action_dim=env.action_dim, 
        is_action_discrete=env.is_action_discrete)

    # construct agent
    agent = Agent(name='sac', 
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
        collect_fn = (
            lambda state, action, reward, done, next_state, **kwargs: 
            replay.add(state=state, action=action, 
            reward=reward, done=done, next_state=next_state))        
        while not replay.good_to_learn():
            run(env, agent.actor, collect_fn)
    else:
        random_sampling(env, replay)

    train(agent, env, replay)
