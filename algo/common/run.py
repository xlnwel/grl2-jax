import numpy as np
from utility.display import pwc


def run(env, agent, *, fn=None, evaluation=False, step=None):
    obs = env.reset()
    for i in range(1, env.max_episode_steps+1):
        action = agent(obs, deterministic=evaluation)
        next_obs, reward, done, _ = env.step(action)
        if fn:
            if step is None:
                fn(obs=obs, action=action, reward=reward, 
                    done=done, next_obs=next_obs)
            else:
                fn(obs=obs, action=action, reward=reward,
                    done=done, next_obs=next_obs, 
                    step=step+i)
        obs = next_obs
        if np.all(done):
            break
        
    return env.score(), env.epslen()

def evaluate(env, agent, n=1, render=False):
    pwc('Evaluation starts', color='cyan')
    scores = []
    epslens = []
    for _ in range(0, n, env.n_envs):
        if hasattr(agent, 'reset_states'):
            agent.reset_states()
        obs = env.reset()
        for _ in range(0, env.max_episode_steps, getattr(env, 'n_ar', 1)):
            if render:
                env.render()
            action = agent(obs, deterministic=True)
            obs, _, done, _ = env.step(action)

            if np.all(done):
                break
        scores.append(env.score())
        epslens.append(env.epslen())

    return scores, epslens