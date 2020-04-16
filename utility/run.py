import numpy as np
from utility.display import pwc


def run(env, agent, step, *, obs=None, fn=None, evaluation=False, nsteps=None):
    if obs is None:
        obs = env.reset()
    nsteps = nsteps or env.max_episode_steps
    for _ in range(1, nsteps+1):
        action = agent(obs, deterministic=evaluation)
        nth_obs, reward, done, _ = env.step(action)
        step += env.n_envs
        if fn:
            fn(env, step, obs=obs, action=action, reward=reward,
                done=done, nth_obs=nth_obs)
        if env.already_done():
            obs = env.reset()
        else:
            obs = nth_obs
        
    return obs, step

def evaluate(env, agent, n=1, record=False):
    pwc('Evaluation starts', color='cyan')
    scores = []
    epslens = []
    imgs = []
    name = env.name
    for _ in range(0, n, env.n_envs):
        if hasattr(agent, 'reset_states'):
            agent.reset_states()
        obs = env.reset()

        for _ in range(0, env.max_episode_steps):
            if record:
                if name.startswith('dm'):
                    imgs.append(obs)
                elif name.startswith('atari'):
                    imgs.append(env.get_screen())
                else:
                    imgs.append(env.render(mode='rgb_array'))
            action = agent(obs, deterministic=True)
            obs, _, done, _ = env.step(action)

            if np.all(env.game_over()):
                break
        scores.append(env.score())
        epslens.append(env.epslen())

    if record:
        return scores, epslens, np.array(imgs, copy=False)
    else:
        return scores, epslens
