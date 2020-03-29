import numpy as np


def run(env, agent, buffer, step, obs):
    for i in range(1, agent.N_STEPS):
        action, logpi, value = agent(obs)
        next_obs, reward, done, _ = env.step(action.numpy())
        mask = env.get_mask()
        buffer.add(obs=obs, 
                    action=action.numpy(), 
                    reward=reward, 
                    value=value.numpy(), 
                    old_logpi=logpi.numpy(), 
                    nonterminal=1-done, 
                    mask=mask)
        obs = next_obs
        step += np.sum(mask)

    _, _, last_value = agent(obs, update_curr_state=False)
    buffer.finish(last_value.numpy())
    agent.learn_log(buffer, step)

    already_done = env.get_already_done()
    idxes = [i for i, d in enumerate(already_done) if d]
    score, epslen = env.get_score(idxes), env.get_epslen(idxes)
    agent.store(score=score, epslen=epslen)

    new_obs = env.reset(idxes)
    for i, o in zip(idxes, new_obs):
        obs[i] = o
    agent.reset_states(reset=already_done)
    buffer.reset()

    return step, obs
