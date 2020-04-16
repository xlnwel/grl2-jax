import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import global_policy

from core.tf_config import *
from utility.utils import Every
from utility.run import run, evaluate
from env.gym_env import create_env
from replay.func import create_replay
from replay.data_pipline import Dataset, process_with_env
from algo.sac.agent import Agent
from algo.sac.nn import create_model


def train(agent, env, eval_env, replay):
    def collect_and_learn(env, step, **kwargs):
        if env.game_over():
            agent.store(score=env.score(), epslen=env.epslen())
        replay.add(**kwargs)
        agent.learn_log(step)

    start_step = agent.global_steps.numpy() + 1
    step = start_step
    obs = None
    collect_fn = lambda *args, **kwargs: replay.add(**kwargs)
    while not replay.good_to_learn():
        obs, step = run(env, env.random_action, step, obs=obs, 
            fn=collect_fn, nsteps=agent.LOG_INTERVAL)

    to_log = Every(agent.LOG_INTERVAL)
    print('Training starts...')
    while step < int(agent.MAX_STEPS):
        obs, step = run(env, agent, step, obs=obs, 
            fn=collect_and_learn, nsteps=agent.LOG_INTERVAL)
        
        if to_log(step):
            eval_score, eval_epslen = evaluate(eval_env, agent)
            agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
            agent.log(step)
            agent.save(steps=step)


def main(env_config, model_config, agent_config, replay_config):
    silence_tf_logs()
    configure_gpu()
    # configure_precision(agent_config.get('precision', 32))

    env = create_env(env_config)
    eval_env_config = env_config.copy()
    eval_env_config['n_envs'] = 10
    eval_env_config['auto_reset'] = False
    # eval_env_config['efficient_envvec'] = True
    eval_env = create_env(eval_env_config)

    replay = create_replay(replay_config)

    dtype = global_policy().compute_dtype
    action_dtype = tf.int32 if env.is_action_discrete else dtype
    data_format = dict(
        obs=((None, *env.obs_shape), dtype),
        action=((None, *env.action_shape), action_dtype),
        reward=((None, ), dtype), 
        nth_obs=((None, *env.obs_shape), dtype),
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

    models = create_model(
        model_config, 
        action_dim=env.action_dim, 
        is_action_discrete=env.is_action_discrete)

    agent = Agent(name='sac', 
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

    # This training process is used for Mujoco tasks, following the same process as OpenAI's spinningup
    # obs = env.reset()
    # epslen = 0
    # for t in range(int(agent.MAX_STEPS)):
    #     if t > 1e4:
    #         action, _ = agent.action(obs)
    #     else:
    #         action = env.random_action()

    #     nth_obs, reward, done, _ = env.step(action)
    #     epslen += 1
    #     done = False if epslen == env.max_episode_steps else done
    #     replay.add(obs=obs, action=action, reward=reward, done=done, nth_obs=nth_obs)
    #     obs = nth_obs

    #     if done or epslen == env.max_episode_steps:
    #         agent.store(score=env.score(), epslen=env.epslen())
    #         obs = env.reset()
    #         epslen = 0

    #     if replay.good_to_learn() and t % 50 == 0:
    #         for _ in range(50):
    #             agent.learn_log()
    #     if (t + 1) % 4000 == 0:
    #         eval_score, eval_epslen = run(eval_env, agent.actor, evaluation=True)

    #         agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
    #         agent.store(**agent.get_value('score', mean=True, std=True, min=True, max=True))
    #         agent.store(**agent.get_value('epslen', mean=True, std=True, min=True, max=True))
    #         agent.store(**agent.get_value('eval_score', mean=True, std=True, min=True, max=True))
    #         agent.store(**agent.get_value('eval_epslen', mean=True, std=True, min=True, max=True))
            
    #         agent.log(step=t)
