import os
import time
import functools
import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import global_policy
import ray

from core.tf_config import configure_gpu, configure_precision, silence_tf_logs
from utility.ray_setup import sigint_shutdown_ray
from utility.graph import video_summary
from utility.utils import Every, TempStore
from utility.run import Runner, evaluate
from env.gym_env import create_env
from replay.func import create_replay
from core.dataset import DataFormat, Dataset, process_with_env
from algo.dreamer.env import make_env
from run import pkg


def train(agent, env, eval_env, replay):
    frame_skip = getattr(env, 'frame_skip', 1)
    buffer = collections.defaultdict(list)
    def collect(env, step, nth_obs,**kwargs):
        for k, v in kwargs.items():
            buffer[k].append(v)
        if env.already_done():
            replay.merge(buffer.copy())
            buffer.clear()
    
    _, step = replay.count_episodes()
    step = max(agent.env_steps, step)

    runner = Runner(env, agent, step=step)
    while not replay.good_to_learn():
        step = runner.run(step_fn=collect)
        
    to_log = Every(agent.LOG_PERIOD)
    to_eval = Every(agent.EVAL_PERIOD)
    print('Training starts...')
    while step < int(agent.MAX_STEPS):
        start_step = step
        start_t = time.time()
        agent.learn_log(step)
        step = runner.run(step_fn=collect, nsteps=agent.TRAIN_PERIOD)
        duration = time.time() - start_t
        agent.store(
            fps=(step-start_step) / duration,
            tps=(agent.N_UPDATES / duration))

        if to_eval(step):
            with TempStore(agent.get_states, agent.reset_states):
                score, epslen, video = evaluate(
                    eval_env, agent, record=True, size=(64, 64))
                video_summary(f'{agent.name}/sim', video, step=step)
                agent.store(eval_score=score, eval_epslen=epslen)
            
        if to_log(step):
            agent.store(fps=(step-start_step)/duration, duration=duration)
            agent.log(step)
            agent.save()

def get_data_format(env, batch_size, sample_size=None):
    dtype = global_policy().compute_dtype
    data_format = dict(
        obs=DataFormat((batch_size, sample_size, *env.obs_shape), dtype),
        action=DataFormat((batch_size, sample_size, *env.action_shape), 'float32'),
        reward=DataFormat((batch_size, sample_size), 'float32'), 
        discount=DataFormat((batch_size, sample_size), 'float32'),
        logpi=DataFormat((batch_size, sample_size), 'float32')
    )
    return data_format

def main(env_config, model_config, agent_config, replay_config):
    silence_tf_logs()
    configure_gpu()
    configure_precision(env_config['precision'])

    use_ray = env_config.get('n_workers', 0) > 1
    if use_ray:
        ray.init()
        sigint_shutdown_ray()

    env = create_env(env_config, make_env)
    eval_env_config = env_config.copy()
    eval_env = create_env(eval_env_config, make_env)

    create_model, Agent = pkg.import_agent(agent_config)
    models = create_model(model_config, env)

    agent = Agent(
        name='dreamer',
        config=agent_config,
        models=models, 
        dataset=None,
        env=env)

    replay_config['dir'] = agent_config['root_dir'].replace('logs', 'data')
    replay = create_replay(replay_config,
        state_keys=list(agent.rssm.state_size._asdict()))
    replay.load_data()
    data_format = get_data_format(env, agent_config['batch_size'], agent_config['sample_size'])
    if agent._store_state:
        data_format.update({
            k: ((agent_config['batch_size'], v), 'float32')
                for k, v in agent.rssm.state_size._asdict().items()
        })
    process = functools.partial(process_with_env, env=env, obs_range=[-.5, .5])
    dataset = Dataset(replay, data_format, process)
    agent.dataset = dataset

    agent.save_config(dict(
        env=env_config,
        model=model_config,
        agent=agent_config,
        replay=replay_config
    ))

    train(agent, env, eval_env, replay)
