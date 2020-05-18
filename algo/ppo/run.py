import numpy as np


def run(env, agent, buffer, step, obs):
    for _ in range(agent.N_STEPS):
        action, terms = agent(obs, reset=env.already_done(), update_rms=True)
        next_obs, reward, done, _ = env.step(action)
        terms['reward'] = agent.normalize_reward(reward)
        buffer.add(action=action,
                    discount=(1-done).astype(np.bool), 
                    **terms)
        obs = next_obs
        step += env.n_envs
        already_done = env.already_done()
        if already_done.any():
            idxes = [i for i, d in enumerate(already_done) if d]
            score, epslen = env.score(idxes), env.epslen(idxes)
            agent.store(score=score, epslen=epslen)
            new_obs = env.reset(idxes)
            for i, o in zip(idxes, new_obs):
                obs[i] = o

    _, terms = agent(obs, update_curr_state=False, reset=env.already_done())
    buffer.finish(terms['value'])
    agent.learn_log(buffer, step)
    buffer.reset()

    return step, obs
