from collections import deque
import numpy as np
from utility.display import pwc


def run(env, agent, step, *, obs=None, fn=None, evaluation=False, nsteps=None):
    if obs is None:
        obs = env.reset()
    nsteps = nsteps or env.max_episode_steps
    for _ in range(1, nsteps+1):
        action = agent(obs, deterministic=evaluation)
        next_obs, reward, done, _ = env.step(action)
        step += env.n_envs
        if fn:
            fn(env, step, obs=obs, action=action, reward=reward,
                done=done, nth_obs=next_obs)
        if env.already_done():
            obs = env.reset()
        else:
            obs = next_obs
        
    return obs, step

def evaluate(env, agent, n=1, record=False):
    pwc('Evaluation starts', color='cyan')
    scores = []
    epslens = []
    m = max(10000, env.max_episode_steps)
    imgs = deque(maxlen=n * m)
    name = env.name
    for _ in range(0, n, env.n_envs):
        if hasattr(agent, 'reset_states'):
            agent.reset_states()
        obs = env.reset()
        for k in range(0, env.max_episode_steps):
            if record:
                if name.startswith('dm'):
                    imgs.append(obs)
                else:
                    imgs.append(env.get_screen())
                
            action = agent(obs, deterministic=True)
            obs, _, done, _ = env.step(action)

            if np.all(env.game_over()):
                break
            if np.any(env.already_done()):
                if env.n_envs == 1:
                    obs = env.reset()
                else:
                    for i, e in enumerate(env.envs):
                        if e.already_done() and not e.game_over():
                            obs[i] = e.reset()

        scores.append(env.score())
        epslens.append(env.epslen())
    
    if record:
        if env.n_envs == 1:
            imgs = np.array(imgs, copy=False)
        else:
            imgs = np.stack(imgs, axis=1)
        return scores, epslens, imgs
    else:
        return scores, epslens
