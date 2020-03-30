import numpy as np


def run(env, agent, buffer, step, *args, **kwargs):
    buffer.reset()
    obs = env.reset()
    agent.reset_states()

    for _ in range(env.max_episode_steps):
        action, logpi, value = agent(obs)
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
        
    _, _, last_value = agent(obs)
    buffer.finish(last_value.numpy())
    agent.learn_log(buffer, step)
    
    score, epslen = env.get_score(), env.get_epslen()
    agent.store(score=score, epslen=epslen)

    return step + np.sum(epslen), _
