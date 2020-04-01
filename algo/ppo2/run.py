import numpy as np


def run(env, agent, buffer, step, obs):
    if isinstance(obs, tuple):
        obs, already_done = obs
    else:
        already_done = env.already_done()
    buffer.store_state(agent.state)
    for _ in range(agent.N_STEPS):
        action, logpi, value = agent(obs, already_done)
        next_obs, reward, done, _ = env.step(action)
        mask = env.mask()
        already_done = env.already_done()
        buffer.add(obs=obs, 
                    action=action, 
                    reward=reward, 
                    value=value, 
                    old_logpi=logpi, 
                    nonterminal=(1-done).astype(np.bool), 
                    mask=mask)
        obs = next_obs
        step += np.sum(mask)
        if already_done.all():
            break

    _, _, last_value = agent(obs, already_done, update_curr_state=False)
    buffer.finish(last_value)
    agent.learn_log(buffer, step)

    idxes = [i for i, d in enumerate(already_done) if d]
    score, epslen = env.score(idxes), env.epslen(idxes)
    agent.store(score=score, epslen=epslen)
    new_obs = env.reset(idxes)
    for i, o in zip(idxes, new_obs):
        obs[i] = o
    buffer.reset()

    return step, obs
