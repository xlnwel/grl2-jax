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
                done=done, next_obs=next_obs)
        obs = next_obs
        if env.already_done():
            obs = env.reset()
        
    return obs, step

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

            if np.all(env.already_done()):
                break
        scores.append(env.score())
        epslens.append(env.epslen())

    return scores, epslens