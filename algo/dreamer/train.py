import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec
import ray

from core.tf_config import configure_gpu, configure_precision, hide_tf_logs
from utility.utils import set_global_seed, Every
from utility.signal import sigint_shutdown_ray
from utility.graph import video_summary
from utility.timer import TBTimer
from env.gym_env import create_env
from replay.func import create_replay
from replay.data_pipline import DataFormat, Dataset
from algo.dreamer.agent import Agent
from algo.dreamer.nn import create_model
from algo.dreamer.env import make_env


def process(data):
    data = data.copy()
    dtype = prec.global_policy().compute_dtype
    with tf.device('cpu:0'):
        data['obs'] = tf.cast(data['obs'], dtype) / 255. - .5

    return data

def run(env, agent, obs=None, already_done=None, 
        fn=None, nsteps=0, evaluation=False):
    if obs is None:
        obs = env.reset()
    if already_done is None:
        already_done = env.get_already_done()
    nsteps = nsteps or env.max_episode_steps
    for i in range(0, nsteps, env.n_ar):
        action = agent(obs, already_done, deterministic=evaluation)
        with TBTimer('env_step', 10000):
            obs, reward, done, info = env.step(action)
        already_done = env.get_already_done()
        if fn:
            fn(already_done, info)
    return obs, already_done, nsteps * env.n_envs * env.n_ar

def train(agent, env, eval_env, replay):
    def collect(already_done, info):
        if already_done.any():
            eps = [info[i]['episode'] for i, d in enumerate(already_done) if d]
            replay.merge(eps)

    def collect_log(already_done, info):
        if already_done.any():
            episodes, scores, epslens = [], [], []
            for i, d in enumerate(already_done):
                if d:
                    eps = info[i]['episode']
                    episodes.append(eps)
                    scores.append(np.sum(eps['reward']))
                    epslens.append(info[i]['n_ar']*(eps['reward'].size-1))
            agent.store(score=scores, epslen=epslens)
            replay.merge(episodes)

    step = agent.global_steps.numpy()

    should_log = Every(agent.LOG_INTERVAL)
    nsteps = agent.TRAIN_INTERVAL // env.n_envs
    obs, already_done = None, None
    while not replay.good_to_learn():
        obs, already_done, _ = run(
            env, env.random_action, obs, already_done, collect)
    print('Training started...')
    while step < int(agent.MAX_STEPS):
        with TBTimer('train'):
            obs, already_done, n = run(
                    env, agent, obs, already_done, collect_log, nsteps)
            step += n
            agent.set_summary_step(step)
            with TBTimer('learn'):
                agent.learn_log(step)

        if should_log(step):
            train_state, train_action = agent.retrieve_states()
            agent.reset_states(None, None)
            with TBTimer('eval'):
                _, _, _ = run(eval_env, agent, evaluation=True)
            with TBTimer('sim'):
                video_summary('dreamer/sim', eval_env.prev_episode['obs'][None], step)
            with TBTimer('log'):
                eval_score = eval_env.get_score()
                eval_epslen = eval_env.get_epslen()
                agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
                # agent.store(**agent.get_value('score', mean=True, std=True, min=True, max=True))
                # agent.store(**agent.get_value('epslen', mean=True, std=True, min=True, max=True))
                # agent.store(**agent.get_value('eval_score', mean=True, std=True, min=True, max=True))
                # agent.store(**agent.get_value('eval_epslen', mean=True, std=True, min=True, max=True))

                agent.log(step)
                agent.save(steps=step)

            agent.reset_states(train_state, train_action)

def main(env_config, model_config, agent_config, 
        replay_config, restore=False, render=False):
    hide_tf_logs()
    set_global_seed(seed=0, tf=tf)
    configure_gpu()
    configure_precision(agent_config['precision'])

    use_ray = env_config.get('n_workers', 0) > 1
    if use_ray:
        ray.init()
        sigint_shutdown_ray()

    env = create_env(env_config, make_env, force_envvec=True)
    eval_env_config = env_config.copy()
    eval_env_config['auto_reset'] = False
    eval_env_config['n_envs'] = 1
    eval_env_config['n_workers'] = 1
    eval_env = create_env(env_config, make_env, force_envvec=True)

    replay_config['dir'] = agent_config['root_dir'].replace('logs', 'data')
    replay = create_replay(replay_config)
    replay.load_data()
    data_format = dict(
        obs=DataFormat((None, *env.obs_shape), env.obs_dtype),
        action=DataFormat((None, *env.action_shape), env.action_dtype),
        reward=DataFormat((None), tf.float32), 
        discount=DataFormat((None), tf.float32),
    )
    print(data_format)
    
    dataset = Dataset(replay, data_format, process, batch_size=agent_config['batch_size'])

    models = create_model(
        model_config, 
        obs_shape=env.obs_shape,
        action_dim=env.action_dim,
        is_action_discrete=env.is_action_discrete
    )

    agent = Agent(name='dreamer',
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
    
    step = 0
    while step < int(agent.MAX_STEPS):
        with TBTimer('learn'):
            agent.learn_log(step)
        step += 1
        with TBTimer('eval'):
            run(eval_env, agent, evaluation=True)
            eval_score = eval_env.get_score()
            eval_epslen = eval_env.get_epslen()
            agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
        agent.log(step)

    train(agent, env, eval_env, replay)
