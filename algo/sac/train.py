from collections import deque
import numpy as np
import ray
import tensorflow as tf

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


LOG_INTERVAL = 4000

def train(agent, env, replay):
    def collect_and_learn(step, **kwargs):
        replay.add(**kwargs)
        if step % 50 == 0:
            for _ in range(50):
                agent.learn_log()

    eval_env = create_gym_env(dict(
        name=env.name, 
        video_path='video',
        log_video=False,
        n_workers=1,
        n_envs=10,
        effective_envvec=True,
        seed=0,
    ))
    start_step = agent.global_steps.numpy() + 1
    scores = deque(maxlen=100)
    epslens = deque(maxlen=100)
    eval_scores = deque(maxlen=10)
    eval_epslens = deque(maxlen=10)
    print('Training started...')
    step = start_step
    log_step = LOG_INTERVAL
    while step < int(agent.MAX_STEPS):
        agent.set_summary_step(step)
        with TBTimer(f'trajectory', agent.TIME_INTERVAL, to_log=agent.timer):
            score, epslen = run(
                env, agent.actor, fn=collect_and_learn, timer=agent.timer, step=step)
        step += epslen
        scores.append(score)
        epslens.append(epslen)
        
        if step > log_step:
            log_step += LOG_INTERVAL
            agent.save(steps=step)

            
            with TBTimer(f'evaluation', agent.TIME_INTERVAL, to_log=agent.timer):
                eval_score, eval_epslen = run(
                    eval_env, agent.actor, evaluation=True, timer=agent.timer, name='eval')
            eval_scores.append(eval_score)
            eval_epslens.append(eval_epslen)
            
            agent.store(
                score=np.mean(scores),
                score_std=np.std(scores),
                score_max=np.max(scores),
                epslen=np.mean(epslens),
                epslen_std=np.std(epslens),
                eval_score=np.mean(eval_scores),
                eval_score_std=np.std(eval_scores),
                eval_score_max=np.max(eval_scores),
                eval_epslen=np.mean(eval_epslens),
                eval_epslen_std=np.std(eval_epslens),
            )
            agent.log(step)


def main(env_config, model_config, agent_config, replay_config, restore=False, render=False):
    set_global_seed(seed=env_config['seed'], tf=tf)
    # tf.debugging.set_log_device_placement(True)
    configure_gpu()

    env = create_gym_env(env_config)

    # construct replay
    replay = create_replay(replay_config)
    data_format = dict(
        state=(env.state_dtype, (None, *env.state_shape)),
        action=(env.action_dtype, (None, *env.action_shape)),
        reward=(tf.float32, (None, )), 
        next_state=(env.state_dtype, (None, *env.state_shape)),
        done=(tf.float32, (None, )),
        steps=(tf.float32, (None, )),
    )
    dataset = Dataset(replay, data_format)

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
        collect_fn = lambda **kwargs: replay.add(**kwargs)      
        while not replay.good_to_learn():
            run(env, agent.actor, collect_fn)
    else:
        random_sampling(env, replay)

    train(agent, env, replay)
