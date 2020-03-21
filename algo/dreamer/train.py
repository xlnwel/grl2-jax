import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

from core.tf_config import configure_gpu, configure_precision
from utility.utils import set_global_seed, Every
from env.gym_env import create_gym_env
from replay.func import create_replay
from replay.data_pipline import DataFormat, Dataset
from algo.run import run, random_sampling
from algo.dreamer.agent import Agent
from algo.dreamer.nn import create_model


def train(agent, env, replay):
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

    print('Training started...')
    step = start_step
    should_log = Every(agent.LOG_INTERVAL)
    should_train = Every(agent.TRAIN_INTERVAL)
    already_done = np.zeros(env.n_envs, np.bool)
    obs = env.reset()
    while step < int(agent.MAX_STEPS):
        agent.set_summary_step(step)
        action = agent(obs, already_done)
        obs, reward, done, info = env.step(action)
        already_done = env.get_already_done()
        if already_done.any():
            idxes = [idx for idx, d in enumerate(already_done) if d]
            for i in idxes:
                eps = info[i]['episode']
                score = np.sum(eps['reward'])
                epslen = eps['reward'].size
                agent.store(score=score, epslen=epslen)
                replay.merge(eps)
                step += epslen
        step += env.n_envs
        if should_train(step):
            agent.learn_log(step)

        if should_log(step):
            agent.save(steps=step)

            # eval_score, eval_epslen = run(eval_env, agent, 
            #     evaluation=True, timer=agent.TIMER, name='eval')
            
            # agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
            agent.store(**agent.get_value('score', mean=True, std=True, min=True, max=True))
            agent.store(**agent.get_value('epslen', mean=True, std=True, min=True, max=True))
            # agent.store(**agent.get_value('eval_score', mean=True, std=True, min=True, max=True))
            # agent.store(**agent.get_value('eval_epslen', mean=True, std=True, min=True, max=True))

            agent.log(step)

def main(env_config, model_config, agent_config, 
        replay_config, restore=False, render=False):
    set_global_seed(seed=env_config['seed'], tf=tf)
    configure_gpu()
    configure_precision(agent_config['precision'])

    env = create_gym_env(env_config)

    replay_config['dir'] = agent_config['root_dir'].replace('logs', 'data')
    replay = create_replay(replay_config)
    replay.load_data()
    data_format = dict(
        obs=DataFormat((None, *env.obs_shape), env.obs_dtype),
        action=DataFormat((None, *env.action_shape), env.action_dtype),
        reward=DataFormat((None), tf.float32), 
        done=DataFormat((None), tf.float32),
    )
    print(data_format)
    dataset = Dataset(replay, data_format, batch_size=agent_config['batch_size'])

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
    if not replay.good_to_learn():
        obs = env.reset()
        action = env.random_action()
        obs, reward, done, info = env.step(action)
        already_done = env.get_already_done()
        if already_done.any():
            idxes = [idx for idx, d in enumerate(already_done) if d]
            for i in idxes:
                eps = info[i]['episode']
                replay.merge(eps)
                
    train(agent, env, replay)
