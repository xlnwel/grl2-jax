import time
import functools
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import global_policy

from core.tf_config import *
from utility.utils import Every, TempStore
from utility.graph import video_summary
from utility.timer import TBTimer
from utility.run import Runner, evaluate
from env.gym_env import create_env
from replay.func import create_replay
from core.dataset import Dataset, process_with_env
from run import pkg


def train(agent, env, eval_env, replay):
    def collect_fn(env, step, nth_obs, **kwargs):
        replay.add(**kwargs)
        if env.already_done():
            replay.clear_temp_buffer()
    
    step = agent.env_steps
    runner = Runner(env, agent, step=step)
    while not replay.good_to_learn():
        step = runner.run(step_fn=collect_fn, nsteps=int(1e4))

    to_log = Every(agent.LOG_PERIOD)
    to_eval = Every(agent.EVAL_PERIOD)
    print('Training starts...')
    while step < int(agent.MAX_STEPS):
        start_step = step
        start_t = time.time()
        agent.learn_log(step)
        step = runner.run(step_fn=collect_fn, nsteps=agent.TRAIN_PERIOD)
        duration = time.time() - start_t
        agent.store(
            fps=(step-start_step) / duration,
            tps=(agent.N_UPDATES / duration))

        if to_eval(step):
            with TempStore(agent.get_states, agent.reset_states):
                eval_score, eval_epslen, video = evaluate(
                    eval_env, agent, record=False, size=(64, 64))
                # video_summary(f'{agent.name}/sim', video, step=step)
                agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
        
        if to_log(step):
            agent.log(step)
            agent.save()
    

def get_data_format(env, batch_size, sample_size=None, 
        is_per=False, store_state=False, state_size=None):
    dtype = global_policy().compute_dtype
    obs_dtype = env.obs_dtype if len(env.obs_shape) == 3 else dtype
    data_format = dict(
        obs=((batch_size, sample_size, *env.obs_shape), obs_dtype),
        action=((batch_size, sample_size, *env.action_shape), tf.int32),
        reward=((batch_size, sample_size), dtype), 
        logpi=((batch_size, sample_size), dtype),
        discount=((batch_size, sample_size), dtype),
    )
    if is_per:
        data_format['IS_ratio'] = ((batch_size), dtype)
        data_format['idxes'] = ((batch_size), tf.int32)
    if store_state:
        data_format['h'] = ((batch_size, state_size), dtype)
        data_format['c'] = ((batch_size, state_size), dtype)

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

    create_model, Agent = pkg.import_agent(agent_config)

    silence_tf_logs()
    configure_gpu()
    configure_precision(agent_config.get('precision', 32))

    env = create_env(env_config)
    assert env.n_envs == 1, \
        f'n_envs({env.n_envs}) > 1 is not supported here as it messes with n-step'
    eval_env_config = env_config.copy()
    eval_env = create_env(eval_env_config)

    models = create_model(model_config, env)

    replay_config['dir'] = agent_config['root_dir'].replace('logs', 'data')
    replay = create_replay(replay_config, state_keys=['h', 'c'])
    data_format = get_data_format(
        env, agent_config['batch_size'], agent_config['sample_size'],
        replay_config['type'].endswith('per'), agent_config['store_state'], 
        model_config['lstm_units'])
    
    process = functools.partial(process_with_env, env=env, obs_range=[0, 1])
    dataset = Dataset(replay, data_format, process_fn=process)

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
