import os
import time
import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import global_policy
import ray

from core.tf_config import configure_gpu, configure_precision, silence_tf_logs
from utility.ray_setup import sigint_shutdown_ray
from utility.graph import video_summary
from utility.utils import Every
from utility.run import evaluate
from env.gym_env import create_env
from replay.func import create_replay
from replay.data_pipline import DataFormat, Dataset, process_with_env
from algo.dreamer.env import make_env
from run import pkg


def run(env, agent, step, obs=None, 
    already_done=None, fn=None, nsteps=0, evaluation=False):
    reset_terms = dict(logpi=0)
    if obs is None:
        obs = env.reset(**reset_terms)
    if already_done is None:
        already_done = env.already_done()
    frame_skip = getattr(env, 'frame_skip', 1)
    frames_per_step = env.n_envs * frame_skip
    nsteps = (nsteps or env.max_episode_steps) // frame_skip
    for _ in range(nsteps):
        action, terms = agent(obs, already_done, deterministic=evaluation)
        obs, reward, done, info = env.step(action, **terms)
        already_done = env.already_done()
        step += frames_per_step
        if fn:
            fn(already_done, info)
        if np.any(already_done):
            idxes = [i for i, d in enumerate(already_done) if d]
            new_obs = env.reset(idxes, **reset_terms)
            for i, o in zip(idxes, new_obs):
                obs[i] = o

    return obs, already_done, step

def train(agent, env, eval_env, replay):
    frame_skip = getattr(env, 'frame_skip', 1)
    def collect_log(already_done, info):
        if np.any(already_done):
            episodes, scores, epslens = [], [], []
            for i, d in enumerate(already_done):
                if d:
                    eps = info[i]['episode']
                    episodes.append(eps)
                    scores.append(np.sum(eps['reward']))
                    epslens.append(frame_skip*(eps['reward'].size-1))
            agent.store(score=scores, epslen=epslens)
            replay.merge(episodes)
    _, step = replay.count_episodes()
    step = max(agent.global_steps.numpy(), step)

    nsteps = agent.TRAIN_INTERVAL
    obs, already_done = None, None
    random_agent = lambda *args, **kwargs: (env.random_action(), dict(logpi=0))
    while not replay.good_to_learn():
        obs, already_done, nstep= run(
            env, random_agent, step, obs, already_done, collect_log)
        
    to_log = Every(agent.LOG_INTERVAL)
    to_eval = Every(agent.EVAL_INTERVAL)
    print('Training starts...')
    start_step = step
    start_t = time.time()
    while step < int(agent.MAX_STEPS):
        agent.learn_log(step)
        obs, already_done, step = run(
            env, agent, step, obs, already_done, collect_log, nsteps)
        duration = time.time() - start_t
        if to_eval(step):
            train_state, train_action = agent.retrieve_states()

            score, epslen, video = evaluate(eval_env, agent, record=True, size=(64, 64))
            video_summary(f'{agent.name}/sim', video, step)
            agent.store(eval_score=score, eval_epslen=epslen)
            
            agent.reset_states(train_state, train_action)
        if to_log(step):
            agent.store(fps=(step-start_step)/duration, duration=duration)
            agent.log(step)
            agent.save(steps=step)

            start_step = step
            start_t = time.time()

def get_data_format(env, batch_size, batch_len=None):
    dtype = global_policy().compute_dtype
    data_format = dict(
        obs=DataFormat((batch_size, batch_len, *env.obs_shape), dtype),
        action=DataFormat((batch_size, batch_len, *env.action_shape), dtype),
        reward=DataFormat((batch_size, batch_len), dtype), 
        discount=DataFormat((batch_size, batch_len), dtype),
        logpi=DataFormat((batch_size, batch_len), dtype)
    )
    return data_format

def main(env_config, model_config, agent_config, replay_config):
    algo = agent_config['algorithm']
    env = env_config['name']
    if 'atari' not in env \
        and 'dmc' not in env:
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
    configure_precision(env_config['precision'])

    use_ray = env_config.get('n_workers', 0) > 1
    if use_ray:
        ray.init()
        sigint_shutdown_ray()

    env = create_env(env_config, make_env, force_envvec=True)
    eval_env_config = env_config.copy()
    eval_env_config['n_envs'] = 1
    eval_env_config['n_workers'] = 1
    eval_env_config['log_episode'] = False
    if 'reward_hack' in eval_env_config:
        del eval_env_config['reward_hack']
    eval_env = create_env(eval_env_config, make_env)

    replay_config['dir'] = agent_config['root_dir'].replace('logs', 'data')
    replay = create_replay(replay_config)
    replay.load_data()
    data_format = get_data_format(env, agent_config['batch_size'], agent_config['batch_len'])
    print(data_format)
    process = functools.partial(process_with_env, env=env, obs_range=[-.5, .5])
    dataset = Dataset(replay, data_format, process)

    create_model, Agent = pkg.import_agent(agent_config)
    models = create_model(
        model_config, 
        obs_shape=env.obs_shape,
        action_dim=env.action_dim,
        is_action_discrete=env.is_action_discrete
    )

    agent = Agent(
        name='dreamer',
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