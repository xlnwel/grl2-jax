import numpy as np


# this training scheme does not work well -_-#
def run_trajectories(env, agent, buffer, learn_freq, epoch):
    buffer.reset()
    obs = env.reset()
    agent.reset_states(batch_size=env.n_envs)
    np.testing.assert_allclose(agent.prev_state, 0.)
    np.testing.assert_allclose(agent.curr_state, 0.)

    for i in range(1, env.max_episode_steps+1):
        action, logpi, value = agent.step(obs)
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
            if buffer.good_to_learn():
                _, _, last_value = agent.step(obs, update_curr_state=False)
                buffer.finish(last_value.numpy())
                agent.learn_log(buffer, epoch)
            break

        if i % learn_freq == 0 and buffer.good_to_learn():
            _, _, last_value = agent.step(obs, update_curr_state=False)
            buffer.finish(last_value.numpy())
            agent.learn_log(buffer, epoch)
            buffer.reset()

    score, epslen = env.get_score(), env.get_epslen()

    return score, epslen
