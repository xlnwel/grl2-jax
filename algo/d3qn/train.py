import time
import functools
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import global_policy

from core.tf_config import *
from utility.utils import Every
from utility.graph import video_summary
from utility.run import run, evaluate
from env.gym_env import create_env
from replay.func import create_replay
from replay.data_pipline import Dataset, process_with_env
from algo.d3qn.agent import Agent
from algo.d3qn.nn import create_model


def train(agent, env, eval_env, replay):
    def collect_and_learn(env, step, **kwargs):
        replay.add(**kwargs)
        if env.game_over():
            agent.store(score=env.score(), epslen=env.epslen())
            # we reset noisy every episode. Theoretically, 
            # this follows the guide of deep exploration, 
            # but its practical effectivness is not verified. 
            # More importantly, it saves time!
            agent.reset_noisy()
        if step % agent.TRAIN_INTERVAL == 0:
            agent.learn_log(step)
    
    start_step = agent.global_steps.numpy()
    step = start_step
    obs = None
    collect_fn = lambda *args, **kwargs: replay.add(**kwargs)
    while not replay.good_to_learn():
        obs, step = run(env, env.random_action, step, obs=obs, 
            fn=collect_fn, nsteps=agent.LOG_INTERVAL)

    to_log = Every(agent.LOG_INTERVAL)
    to_eval = Every(agent.EVAL_INTERVAL)
    print('Training starts...')
    while step < int(agent.MAX_STEPS):
        start = time.time()
        obs, step = run(env, agent, step, obs=obs,
            fn=collect_and_learn, nsteps=agent.LOG_INTERVAL)
        
        agent.store(fps=agent.LOG_INTERVAL / (time.time() - start))
        if to_eval(step):
            eval_score, eval_epslen, video = evaluate(eval_env, agent, record=True)
            video_summary('d3qn/sim', video, step)
            agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
        agent.log(step)
        agent.save(steps=step)
    

def get_data_format(env, replay_config):
    dtype = global_policy().compute_dtype
    obs_dtype = env.obs_dtype if len(env.obs_shape) == 3 else dtype
    data_format = dict(
        obs=((None, *env.obs_shape), obs_dtype),
        action=((None, *env.action_shape), tf.int32),
        reward=((None, ), dtype), 
        nth_obs=((None, *env.obs_shape), obs_dtype),
        done=((None, ), dtype),
    )
    if replay_config['type'].endswith('proportional'):
        data_format['IS_ratio'] = ((None, ), dtype)
        data_format['saved_idxes'] = ((None, ), tf.int32)
    if replay_config.get('n_steps', 1) > 1:
        data_format['steps'] = ((None, ), dtype)

    return data_format

def main(env_config, model_config, agent_config, replay_config):
    silence_tf_logs()
    configure_gpu()
    configure_precision(agent_config.get('precision', 32))

    env = create_env(env_config)
    eval_env_config = env_config.copy()
    eval_env_config['n_envs'] = 1
    eval_env = create_env(eval_env_config)
    replay = create_replay(replay_config)

    data_format = get_data_format(env, replay_config)
    print(data_format)
    process = functools.partial(process_with_env, env=env)
    dataset = Dataset(replay, data_format, process_fn=process)
    # construct models
    models = create_model(model_config, env.action_dim)

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

    train(agent, env, eval_env, replay)