import numpy as np
import tensorflow as tf


def run_trajectories(env, agent, buffer):
    buffer.reset()
    agent.reset_states()
    obs = env.reset()

    for _ in range(env.max_episode_steps):
        action, logpi, value = agent.step(tf.convert_to_tensor(obs, tf.float32))
        next_obs, reward, done, _ = env.step(action.numpy())
        buffer.add(obs=obs, 
                    action=action.numpy(), 
                    reward=reward, 
                    value=value.numpy(), 
                    old_logpi=logpi.numpy(), 
                    nonterminal=1-done, 
                    mask=env.get_mask())
        
        obs = next_obs
        if np.all(done):
            break
        
    _, _, last_value = agent.step(tf.convert_to_tensor(obs, tf.float32))
    buffer.finish(last_value.numpy())

    score, epslen = env.get_score(), env.get_epslen()

    return score, epslen
