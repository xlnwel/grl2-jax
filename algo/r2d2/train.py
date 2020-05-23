import time
import functools
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import global_policy

from core.tf_config import *
from utility.utils import Every
from utility.graph import video_summary
from utility.timer import TBTimer
from utility.run import run, evaluate
from env.gym_env import create_env
from replay.func import create_replay
from core.dataset import Dataset, process_with_env
from algo.d3qn.agent import Agent
from algo.d3qn.nn import create_model


def train(agent, env, eval_env, replay):
    def collect_fn(*args, **kwargs):
        replay.add(**kwargs)

    def collect_and_learn(env, step, **kwargs):
        replay.add(**kwargs)
        if env.game_over():
            agent.store(score=env.score(), epslen=env.epslen())
            # we reset noisy every episode. Theoretically, 
            # this follows the guide of deep exploration.
            # More importantly, it saves time!
            agent.reset_noisy()
        if step % agent.TRAIN_PERIOD == 0:
            agent.learn_log(step)
    
    start_step = agent.env_steps
    step = start_step
    obs = None
    while not replay.good_to_learn():
        obs, step = run(env, env.random_action, step, obs=obs, 
            fn=collect_fn, nsteps=agent.LOG_PERIOD)

    to_log = Every(agent.LOG_PERIOD)
    to_eval = Every(agent.EVAL_PERIOD)
    print('Training starts...')
    while step < int(agent.MAX_STEPS):
        start = time.time()
        start_step = step
        obs, step = run(env, agent, step, obs=obs,
            fn=collect_and_learn, nsteps=agent.LOG_PERIOD)
        
        agent.store(fps=(step - start_step) / (time.time() - start))
        if to_eval(step):
            eval_score, eval_epslen, video = evaluate(
                eval_env, agent, record=True, size=(64, 64))
            video_summary(f'{agent.name}/sim', video, step=step)
            agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
        with TBTimer('log', 10):
            agent.log(step)
        with TBTimer('save', 10):
            agent.save()
    

def get_data_format(env, replay_config):
    dtype = global_policy().compute_dtype
    obs_dtype = env.obs_dtype if len(env.obs_shape) == 3 else dtype
    data_format = dict(
        obs=((None, *env.obs_shape), obs_dtype),
        action=((None, *env.action_shape), tf.int32),
        reward=((None, ), dtype), 
        nth_obs=((None, *env.obs_shape), obs_dtype),
        discount=((None, ), dtype),
    )
    if replay_config['type'].endswith('per'):
        data_format['IS_ratio'] = ((None, ), dtype)
        data_format['idxes'] = ((None, ), tf.int32)
    if replay_config.get('n_steps', 1) > 1:
        data_format['steps'] = ((None, ), dtype)

    return data_format

def main(env_config, model_config, agent_config, replay_config):
    algo = agent_config['algorithm']
    env = env_config['name']
    if 'atari' not in env:
        from run.pkg import get_package
        from utility import yaml_op
        root_dir = agent_config['root_dir']
        model_name = agent_config['model_name']
        directory = get_package(algo, 0, '/')
        config = yaml_op.load_config(f'{directory}/config2.yaml')
        env_config = config['env']
        model_config = config['model']
        agent_config = config['agent']
        replay_config = config['replay']
        agent_config['root_dir'] = root_dir
        agent_config['model_name'] = model_name
        env_config['name'] = env

    silence_tf_logs()
    configure_gpu()
    configure_precision(agent_config.get('precision', 32))

    env = create_env(env_config)
    assert env.n_envs == 1, \
        f'n_envs({env.n_envs}) > 1 is not supported here as it messes with n-step'
    eval_env_config = env_config.copy()
    eval_env = create_env(eval_env_config)
    replay = create_replay(replay_config)

    data_format = get_data_format(env, replay_config)
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
