import time
import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import global_policy

from core.tf_config import *
from utility.graph import video_summary
from utility.run import Runner, evaluate
from utility import pkg
from env.func import create_env
from replay.func import create_replay
from core.dataset import Dataset


def train(agent, env, eval_env, replay):
    def collect_and_learn(env, step, reset, **kwargs):
        if reset:
            kwargs['next_obs'] = env.prev_obs()
        replay.add(**kwargs)
        agent.train_record(step)

    start_step = agent.env_step
    step = start_step
    collect = lambda *args, **kwargs: replay.add(**kwargs)
    runner = Runner(env, agent, step=step, nsteps=agent.LOG_PERIOD)
    while not replay.good_to_learn():
        step = runner.run(
            action_selector=env.random_action,
            step_fn=collect)

    print('Training starts...')
    while step < int(agent.MAX_STEPS):
        start_step = step
        eps = np.sum(runner.episodes)
        start = time.time()
        step = runner.run(step_fn=collect_and_learn)
        agent.store(
            fps=(step - start_step) / (time.time() - start),
            tpe=(time.time() - start) / (np.sum(runner.episodes) - eps))

        eval_score, eval_epslen, video = evaluate(
            eval_env, agent, record=agent.RECORD, size=(64, 64))
        if agent.RECORD:
            video_summary(f'{agent.name}/sim', video, fps=20, step=step)
        agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
        agent.record(step=step)
        agent.save()

def random_crop(imgs, output_shape):
    if len(output_shape) == 2:
        output_shape = output_shape + imgs.shape[-1:]
    fn = lambda x: tf.image.random_crop(x, output_shape)
    cropped_imgs = tf.map_fn(fn, imgs)
    return cropped_imgs

def process_with_env(data, env, cropped_obs_shape, one_hot_action=True, dtype=tf.float32):
    with tf.device('cpu:0'):
        obs = data['obs']
        data['obs'] = random_crop(obs, cropped_obs_shape)
        data['next_obs'] = random_crop(data['next_obs'], cropped_obs_shape)
        data['obs_pos'] = random_crop(obs, cropped_obs_shape)
        if env.is_action_discrete and one_hot_action:
            data['action'] = tf.one_hot(data['action'], env.action_dim, dtype=dtype)
    return data

def main(env_config, model_config, agent_config, replay_config):
    silence_tf_logs()
    configure_gpu()

    env = create_env(env_config)
    assert env.n_envs == 1, \
        f'n_envs({env.n_envs}) > 1 is not supported here as it messes with n-step'
    eval_env_config = env_config.copy()
    eval_env = create_env(eval_env_config)

    replay = create_replay(replay_config)

    am = pkg.import_module('agent', config=agent_config)
    data_format = am.get_data_format(
        env=env, 
        is_per=replay_config['replay_type'].endswith('per'), 
        n_steps=replay_config['n_steps'])
    process = functools.partial(process_with_env, env=env, 
        cropped_obs_shape=tuple(agent_config['obs_shape']))
    dataset = Dataset(replay, data_format, process_fn=process)

    create_model, Agent = pkg.import_agent(config=agent_config)
    models = create_model(model_config, env)
    agent = Agent(
        name=env.name,
        config=agent_config, 
        models=models, 
        dataset=dataset, 
        env=env)
    # from algo.curl.agent2 import CurlSacAgent
    # agent = CurlSacAgent(name=env.name, config=config, env=env, replay=replay)
    
    agent.save_config(dict(
        env=env_config,
        model=model_config,
        agent=agent_config,
        replay=replay_config
    ))
    
    train(agent, env, eval_env, replay)
