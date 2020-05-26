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
from utility.run import Runner, evaluate
from env.gym_env import create_env
from replay.func import create_replay
from core.dataset import Dataset, process_with_env
from run import pkg


def train(agent, env, eval_env, replay):
    def collect_and_learn(env, step, **kwargs):
        replay.add(**kwargs)
        if env.game_over():
            # we reset noisy every episode. Theoretically, 
            # this follows the guide of deep exploration.
            # More importantly, it saves time!
            agent.reset_noisy()
        if step % agent.TRAIN_PERIOD == 0:
            agent.learn_log(step)
    
    step = agent.env_steps
    collect = lambda *args, **kwargs: replay.add(**kwargs)
    runner = Runner(env, agent, step=step)
    while not replay.good_to_learn():
        step = runner.run(
            action_selector=env.random_action, 
            step_fn=collect, nsteps=int(1e4))

    to_log = Every(agent.LOG_PERIOD)
    to_eval = Every(agent.EVAL_PERIOD)
    print('Training starts...')
    while step < int(agent.MAX_STEPS):
        start_step = step
        start = time.time()
        step = runner.run(step_fn=collect_and_learn, nsteps=agent.LOG_PERIOD)
        agent.store(fps=(step - start_step) / (time.time() - start))
        
        if to_eval(step):
            eval_score, eval_epslen, video = evaluate(
                eval_env, agent, record=False, size=(64, 64))
            # video_summary(f'{agent.name}/sim', video, step=step)
            agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
        action = agent.get_raw_value('action')
        agent.histogram_summary({'action': action}, step=step)
        agent.log(step)
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
        print('Any changes to config is dropped as we switch to a non-atari environment')
        from run.pkg import get_package
        from utility import yaml_op
        root_dir = agent_config['root_dir']
        model_name = agent_config['model_name']
        directory = pkg.get_package(algo, 0, '/')
        config = yaml_op.load_config(f'{directory}/config2.yaml')
        env_config = config['env']
        model_config = config['model']
        agent_config = config['agent']
        replay_config = config['replay']
        agent_config['root_dir'] = root_dir
        agent_config['model_name'] = model_name
        env_config['name'] = env

    create_model, Agent = pkg.import_agent(agent_config)
    
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
    models = create_model(model_config, env)

    # construct agent
    agent = Agent(
        name='q',
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
