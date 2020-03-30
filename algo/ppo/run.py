import numpy as np


def run(env, agent, buffer, step, obs):
    for _ in range(agent.N_STEPS):
        action, logpi, value = agent(obs)
        next_obs, reward, done, _ = env.step(action.numpy())
        buffer.add(obs=obs, 
                    action=action.numpy(), 
                    reward=reward, 
                    value=value.numpy(), 
                    old_logpi=logpi.numpy(), 
                    nonterminal=1-done)
        obs = next_obs
        step += env.n_envs
        already_done = env.get_already_done()
        if already_done.any():
            idxes = [i for i, d in enumerate(already_done) if d]
            score, epslen = env.get_score(idxes), env.get_epslen(idxes)
            agent.store(score=score, epslen=epslen)
            new_obs = env.reset(idxes)
            for i, o in zip(idxes, new_obs):
                obs[i] = o

    _, _, last_value = agent(obs)
    buffer.finish(last_value.numpy())
    agent.learn_log(buffer, step)
    buffer.reset()

    return step, obs
