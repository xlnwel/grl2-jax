import numpy as np
import tensorflow as tf

    
def run_trajectory(env, actor, fn=None):
    """ Sample trajectories
    Args:
        actor: the model responsible for taking actions
        fn: a function that specifies what to do after each env step
    """
    state = env.reset()

    for _ in range(env.max_episode_steps):
        state = np.expand_dims(state, 0)
        action = actor.step(tf.convert_to_tensor(state, tf.float32))
        next_state, reward, done, _ = env.step(action.numpy())
        if fn:
            fn(state, action, reward, done)
        state = next_state
        if done:
            break
        
    return env.get_score(), env.get_epslen()

def run_trajectories(env, actor, fn=None):
    """ Sample trajectories
    Args:
        actor: the model responsible for taking actions
        fn: a function that specifies what to do after each env step
    """
    state = env.reset()

    for _ in range(env.max_episode_steps):

        action = actor.step(tf.convert_to_tensor(state, tf.float32))
        next_state, reward, done, _ = env.step(action.numpy())
        if fn:
            fn(state, action, reward, done)
        state = next_state
        if np.all(done):
            break

    return env.get_score(), env.get_epslen()

def random_sampling(env, buffer):
    """ Interact with the environment with random actions to 
    collect data for buffer initialization 
    """
    state = env.reset()

    while not buffer.good_to_learn:
        for _ in range(env.max_episode_steps):
            action = env.random_action()
            next_state, reward, done, _ = env.step(action)
            buffer.add(state, action, reward, done)
            state = next_state
            if done:
                break
        