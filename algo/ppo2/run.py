import numpy as np
import tensorflow as tf


def run_trajectories(env, agent, buffer, learn_freq, epoch):
    buffer.reset()
    state = env.reset()
    agent.reset_states()
    np.testing.assert_allclose(agent.prev_states, 0.)
    np.testing.assert_allclose(agent.curr_states, 0.)

    for i in range(1, env.max_episode_steps+1):
        action, logpi, value = agent.step(state)
        next_state, reward, done, _ = env.step(action.numpy())
        buffer.add(state=state, 
                    action=action.numpy(), 
                    reward=np.expand_dims(reward, 1), 
                    value=value.numpy(), 
                    old_logpi=logpi.numpy(), 
                    nonterminal=np.expand_dims(1-done, 1), 
                    mask=np.expand_dims(env.get_mask(), 1))
        state = next_state
        if np.all(done):
            _, _, last_value = agent.step(state, update_curr_states=False)
            buffer.finish(last_value.numpy())
            agent.learn_log(buffer, epoch)
            break

        if i % learn_freq == 0:
            _, _, last_value = agent.step(state, update_curr_states=False)
            buffer.finish(last_value.numpy())
            agent.learn_log(buffer, epoch)
            buffer.reset()

    score, epslen = env.get_score(), env.get_epslen()

    return score, epslen
