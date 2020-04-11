import functools
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import global_policy

from core.tf_config import *
from utility.utils import Every
from env.gym_env import create_env
from replay.func import create_replay
from replay.data_pipline import Dataset, process_with_env
from algo.common.run import run, evaluate
from algo.d3qn.agent import Agent
from algo.d3qn.nn import create_model


def train(agent, env, eval_env, replay):
    def collect_and_learn(env, step, **kwargs):
        replay.add(**kwargs)
        if env.already_done():
            agent.store(score=env.score(), epslen=env.epslen())
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
        # if to_eval(step):
        #     eval_score, eval_epslen = evaluate(eval_env, agent)
        #     agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
        # agent.q.reset_noisy()
        obs, step = run(env, agent, step, obs=obs,
            fn=collect_and_learn, nsteps=agent.LOG_INTERVAL)
        
        if to_eval(step):
            eval_score, eval_epslen = evaluate(eval_env, agent)
            agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
        if to_log(step):
            agent.log(step)
            agent.save(steps=step)
    

def main(env_config, model_config, agent_config, replay_config, restore=False, render=False):
    silence_tf_logs()
    configure_gpu()
    configure_precision(agent_config.get('precision', 32))

    env = create_env(env_config)
    eval_env_config = env_config.copy()
    eval_env_config['n_envs'] = 1
    eval_env_config['auto_reset'] = False
    # eval_env_config['efficient_envvec'] = True
    eval_env = create_env(eval_env_config)
    replay = create_replay(replay_config)

    dtype = global_policy().compute_dtype
    data_format = dict(
        obs=((None, *env.obs_shape), dtype),
        action=((None, *env.action_shape), tf.int32),
        reward=((None, ), dtype), 
        next_obs=((None, *env.obs_shape), dtype),
        done=((None, ), dtype),
    )
    if replay_config['type'].endswith('proportional'):
        data_format['IS_ratio'] = ((None, ), dtype)
        data_format['saved_idxes'] = ((None, ), tf.int32)
    if replay_config.get('n_steps', 1) > 1:
        data_format['steps'] = ((None, ), dtype)
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
