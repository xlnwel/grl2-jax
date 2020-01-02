import numpy as np
import tensorflow as tf

from utility.timer import TBTimer
from env.gym_env import Env, EnvVec, EfficientEnvVec
from env.wrappers import get_wrapper_by_name


LOG_INTERVAL = 10000

def random_sampling(env, buffer):
    """ Interact with the environment with random actions to 
    collect data for buffer initialization 
    """
    while not buffer.good_to_learn():
        state = env.reset()
        for _ in range(env.max_episode_steps):
            action = env.random_action()
            next_state, reward, done, _ = env.step(action)
            buffer.add(state=state, action=action, reward=reward, done=done, next_state=next_state)
            state = next_state
            if done:
                break

def run(env, actor, *, fn=None, evaluation=False, timer=False, name='run', **kwargs):
    if isinstance(env, Env):
        return run_trajectory(env, actor, fn=fn, 
            evaluation=evaluation, timer=timer, name=name, **kwargs)
    elif isinstance(env, EnvVec):
        return run_trajectories1(env, actor, fn=fn, 
            evaluation=evaluation, timer=timer, name=name)
    elif isinstance (env, EfficientEnvVec):
        return run_trajectories2(env, actor, fn=fn, 
            evaluation=evaluation, timer=timer, name=name)

def run_trajectory(env, actor, *, fn=None, evaluation=False, 
                    timer=False, step=None, render=False, name='traj'):
    """ Sample a trajectory

    Args:
        env: an Env instance
        actor: the model responsible for taking actions
        fn: a function that specifies what to do after each env step
        step: environment step
    """
    action_fn = actor.det_action if evaluation else actor.action

    while True:
        state = env.reset()
        for i in range(1, env.max_episode_steps+1):
            if render:
                env.render()
            state_expanded = np.expand_dims(state, 0)
            with TBTimer(f'{name} agent_step', LOG_INTERVAL, to_log=timer):
                action = action_fn(tf.convert_to_tensor(state_expanded, tf.float32)).numpy()[0]
            with TBTimer(f'{name} env_step', LOG_INTERVAL, to_log=timer):
                next_state, reward, done, _ = env.step(action)
            if fn:
                if step is None:
                    fn(state=state, action=action, reward=reward, 
                        done=done, next_state=next_state)
                else:
                    fn(state=state, action=action, reward=reward,
                        done=done, next_state=next_state, step=step+i)
            state = next_state
            if done:
                break
        # test the effectiveness of Atari wrappers
        # print(env.env.lives, env.get_score(), env.get_epslen())
        if env.already_done:
            break
        else:
            print(f'not already done, {env.get_epslen()}')
        
    return env.get_score(), env.get_epslen()

def run_trajectories1(envvec, actor, fn=None, evaluation=False, timer=False, name='trajs'):
    """ Sample trajectories

    Args:
        envvec: an EnvVec instance
        actor: the model responsible for taking actions
        fn: a function that specifies what to do after each env step
    """
    state = envvec.reset()
    action_fn = actor.det_action if evaluation else actor.action

    for _ in range(envvec.max_episode_steps):
        with TBTimer(f'{name} agent_step', LOG_INTERVAL, to_log=timer):
            action = action_fn(tf.convert_to_tensor(state, tf.float32))
        with TBTimer(f'{name} env_step', LOG_INTERVAL, to_log=timer):
            next_state, reward, done, _ = envvec.step(action.numpy())
        if fn:
            fn(state=state, action=action, reward=reward, done=done, 
                next_state=next_state, mask=envvec.get_mask())
        state = next_state
        if np.all(done):
            break

    return envvec.get_score(), envvec.get_epslen()

def run_trajectories2(envvec, actor, fn=None, evaluation=False, timer=False, name='trajs'):
    """ Sample trajectories

    Args:
        envvec: an EfficientEnvVec instance
        actor: the model responsible for taking actions
        fn: a function that specifies what to do after each env step
    """
    state = envvec.reset()
    action_fn = actor.det_action if evaluation else actor.action

    for _ in range(envvec.max_episode_steps):
        with TBTimer(f'{name} agent_step', LOG_INTERVAL, to_log=timer):
            action = action_fn(tf.convert_to_tensor(state, tf.float32))
        with TBTimer(f'{name} env_step', LOG_INTERVAL, to_log=timer):
            next_state, reward, done, info = envvec.step(action.numpy())
        if fn:
            env_ids = [i['env_id'] for i in info]
            fn(state=state, action=action, reward=reward, done=done, 
                next_state=next_state, mask=envvec.get_mask(), env_ids=env_ids)
        state = next_state[(1 - done).astype(np.bool)]
        if np.all(done):
            break

    return envvec.get_score(), envvec.get_epslen()

def run_steps(envvec, actor, state, steps=None, fn=None, evaluation=False, timer=False, name='steps'):
    # assume the envvec is resetted before, 
    # and step will automatically reset at done
    assert get_wrapper_by_name(envvec, 'AutoReset')
    steps = steps or envvec.max_episode_steps
    action_fn = actor.det_action if evaluation else actor.action

    scores, epslens = [], []
    for _ in range(steps):
        with TBTimer(f'{name} agent_step', LOG_INTERVAL, to_log=timer):
            action = action_fn(tf.convert_to_tensor(state, tf.float32))
        with TBTimer(f'{name} env_step', LOG_INTERVAL, to_log=timer):
            next_state, reward, done, _ = envvec.step(action.numpy())
        if np.any(done):
            scores.append(envvec.get_score()[done])
            epslens.append(envvec.get_epslen()[done])
        if fn:
            to_return = fn(state=state, action=action, reward=reward, done=done, 
                next_state=next_state, mask=envvec.get_mask())
            if to_return:
                break
                
    return scores or None, epslens or None
