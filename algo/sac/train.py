from collections import deque
import numpy as np
import ray
import tensorflow as tf

from utility.utils import set_global_seed
from core.tf_config import configure_gpu
from utility.signal import sigint_shutdown_ray
from utility.timer import TBTimer
from utility.utils import step_str
from env.gym_env import create_gym_env
from replay.func import create_replay
from replay.data_pipline import Dataset
from algo.run import run, random_sampling
from algo.sac.agent import Agent
from algo.sac.nn import create_model


LOG_INTERVAL = 4000

def train(agent, env, replay):
    def collect_and_learn(step, **kwargs):
        replay.add(**kwargs)
        if step % 50 == 0:
            for _ in range(50):
                agent.learn_log()

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
    log_step = LOG_INTERVAL
    while step < int(agent.MAX_STEPS):
        agent.set_summary_step(step)
        with TBTimer(f'trajectory', agent.TIME_INTERVAL, to_log=agent.timer):
            score, epslen = run(
                env, agent.actor, fn=collect_and_learn, timer=agent.timer, step=step)
        agent.store(score=env.get_score(), epslen=env.get_epslen())
        step += epslen
        
        if step > log_step:
            log_step += LOG_INTERVAL
            agent.save(steps=step)

            
            with TBTimer(f'evaluation', agent.TIME_INTERVAL, to_log=agent.timer):
                eval_score, eval_epslen = run(
                    eval_env, agent.actor, evaluation=True, timer=agent.timer, name='eval')
            
            agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
            agent.store(**agent.get_value('score', mean=True, std=True, min=True, max=True))
            agent.store(**agent.get_value('epslen', mean=True, std=True, min=True, max=True))
            agent.store(**agent.get_value('eval_score', mean=True, std=True, min=True, max=True))
            agent.store(**agent.get_value('eval_epslen', mean=True, std=True, min=True, max=True))

            agent.log(step)


def main(env_config, model_config, agent_config, replay_config, restore=False, render=False):
    set_global_seed(seed=env_config['seed'], tf=tf)
    # tf.debugging.set_log_device_placement(True)
    configure_gpu()

    env = create_gym_env(env_config)

    # construct replay
    replay = create_replay(replay_config)
    data_format = dict(
        state=(env.state_dtype, (None, *env.state_shape)),
        action=(env.action_dtype, (None, *env.action_shape)),
        reward=(tf.float32, (None, )), 
        next_state=(env.state_dtype, (None, *env.state_shape)),
        done=(tf.float32, (None, )),
        steps=(tf.float32, (None, )),
    )
    dataset = Dataset(replay, data_format)

    # construct models
    models = create_model(
        model_config, 
        state_shape=env.state_shape, 
        action_dim=env.action_dim, 
        is_action_discrete=env.is_action_discrete)

    # construct agent
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

    state = env.reset()
    eval_env = create_gym_env(dict(
        name=env.name, 
        video_path='video',
        log_video=False,
        n_workers=1,
        n_envs=10,
        effective_envvec=True,
        seed=0,
    ))

    for t in range(int(agent.MAX_STEPS)):
        if t > 1e4:
            action, _ = agent.action(state)
        else:
            action = env.random_action()

        next_state, reward, done, _ = env.step(action)
        replay.add(state=state, action=action, reward=reward, done=done, next_state=next_state)
        state = next_state

        if done:
            agent.store(score=env.get_score(), epslen=env.get_epslen())
            state = env.reset()

        if replay.good_to_learn() and t % 50 == 0:
            for _ in range(50):
                agent.learn_log()
        if (t + 1) % 4000 == 0:
            eval_score, eval_epslen = run(eval_env, agent.actor, evaluation=True)

            agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
            agent.store(**agent.get_value('score', mean=True, std=True, min=True, max=True))
            agent.store(**agent.get_value('epslen', mean=True, std=True, min=True, max=True))
            agent.store(**agent.get_value('eval_score', mean=True, std=True, min=True, max=True))
            agent.store(**agent.get_value('eval_epslen', mean=True, std=True, min=True, max=True))
            
            agent.log(step=t)


    # if restore:
    #     agent.restore()
    #     collect_fn = lambda **kwargs: replay.add(**kwargs)      
    #     while not replay.good_to_learn():
    #         run(env, agent.actor, collect_fn)
    # else:
    #     random_sampling(env, replay)

    # train(agent, env, replay)
