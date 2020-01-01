import numpy as np
import tensorflow as tf

from utility.timer import TBTimer
from env.gym_env import Env, EnvVec, EfficientEnvVec
from env.wrappers import get_wrapper_by_name


LOG_INTERVAL = 10000

def random_sampling(env, buffer, max_ar):
    """ Interact with the environment with random actions to 
    collect data for buffer initialization 
    """
    while not buffer.good_to_learn():
        state = env.reset()
        for _ in range(env.max_episode_steps):
            action = env.random_action()
            n_ar = np.random.randint(1, max_ar+1)
            next_state, reward, done, _ = env.step(action, n_ar=n_ar)
            buffer.add(state=state, action=action, reward=reward, done=done, 
                next_state=next_state, n_ar=n_ar-1)
            state = next_state
            if done:
                break

def run(env, actor, *, fn=None, evaluation=False, timer=False, **kwargs):
    assert get_wrapper_by_name(env.env, 'ActionRepetition') is not None

    if isinstance(env, Env):
        return run_trajectory(env, actor, fn=fn, 
            evaluation=evaluation, timer=timer, **kwargs)
    elif isinstance(env, EnvVec):
        return run_trajectories1(env, actor, fn=fn, 
            evaluation=evaluation, timer=timer)
    elif isinstance (env, EfficientEnvVec):
        return run_trajectories2(env, actor, fn=fn, 
            evaluation=evaluation, timer=timer)

def run_trajectory(env, actor, *, fn=None, evaluation=False, 
                    timer=False, step=None, render=False):
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
            with TBTimer('agent_step', LOG_INTERVAL, to_log=timer):
                action, n_ar = action_fn(tf.convert_to_tensor(state_expanded, tf.float32))
            action = action.numpy()[0]
            n_ar = n_ar.numpy()
            with TBTimer('env_step', LOG_INTERVAL, to_log=timer):
                next_state, reward, done, info = env.step(action, n_ar=n_ar+1)
                n_ar = info['n_ar'] - 1
            if fn:
                if step is None:
                    fn(state=state, action=action, reward=reward, 
                        done=done, next_state=next_state, n_ar=n_ar)
                else:
                    fn(state=state, action=action, reward=reward,
                        done=done, next_state=next_state, n_ar=n_ar, step=step+i)
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

def run_trajectories1(envvec, actor, fn=None, evaluation=False, timer=False):
    """ Sample trajectories

    Args:
        envvec: an EnvVec instance
        actor: the model responsible for taking actions
        fn: a function that specifies what to do after each env step
    """
    state = envvec.reset()
    action_fn = actor.det_action if evaluation else actor.action

    for _ in range(envvec.max_episode_steps):
        with TBTimer('agent_step', LOG_INTERVAL, to_log=timer):
            action, n_ar = action_fn(tf.convert_to_tensor(state, tf.float32))
        with TBTimer('env_step', LOG_INTERVAL, to_log=timer):
            next_state, reward, done, info = envvec.step(action, n_ar=n_ar+1)
            n_ar = np.array([i['n_ar'] for i in info]) - 1
        if fn:
            fn(state=state, action=action, reward=reward, done=done, 
                next_state=next_state, mask=envvec.get_mask(), n_ar=n_ar)
        state = next_state
        if np.all(done):
            break

    return envvec.get_score(), envvec.get_epslen()
