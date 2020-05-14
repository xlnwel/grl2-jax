from collections import deque
import numpy as np
from utility.display import pwc
from utility.timer import TBTimer


def run(env, agent, step, *, obs=None, fn=None, nsteps=None):
    """ run `nstep` agent steps, auto reset if an episodes is done """
    if obs is None:
        obs = env.reset()
    frame_skip = getattr(env, 'frame_skip', 1)
    frames_per_step = env.n_envs * frame_skip
    nsteps = (nsteps or env.max_episode_steps) // frame_skip
    for t in range(nsteps):
        action = agent(obs, deterministic=False)
        next_obs, reward, done, _ = env.step(action)
        step += frames_per_step
        if fn:
            fn(env, step, obs=obs, action=action, reward=reward,
                discount=1-done, nth_obs=next_obs)
        if np.any(env.already_done()):
            if env.n_envs == 1:
                obs = env.reset()
            else:
                for i, e in enumerate(env.envs):
                    if e.already_done():
                        obs[i] = e.reset()
        else:
            obs = next_obs
        
    return obs, step

def run_traj(env, agent, step, *, fn=None):
    # omit inputs 'obs' and 'nsteps'
    obs = env.reset()
    while True:
        action = agent(obs, deterministic=False)
        next_obs, reward, done, _ = env.step(action)
        if fn:
            if env.n_envs == 1:
                fn(env, step, obs=obs, action=action, reward=reward,
                    discount=1-done, nth_obs=next_obs)
            else:
                fn(env, step, obs=obs, action=action, reward=reward,
                    discount=1-done, nth_obs=next_obs, mask=env.mask())
        if np.all(env.game_over()):
            break
        else:
            obs = next_obs

def evaluate(env, agent, n=1, record=False, size=None, video_len=1000):
    pwc('Evaluation starts', color='cyan')
    scores = []
    epslens = []
    maxlen = min(video_len, env.max_episode_steps)
    imgs = deque(maxlen=maxlen)
    name = env.name
    for _ in range(0, n, env.n_envs):
        if hasattr(agent, 'reset_states'):
            agent.reset_states()
        obs = env.reset()
        while True:
            if record:
                if name.startswith('dm'):
                    imgs.append(obs)
                else:
                    imgs.append(env.get_screen(size=size))
            action = agent(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            if np.any(env.already_done()):
                if env.n_envs == 1:
                    if env.game_over():
                        break
                    else:
                        obs = env.reset()
                else:
                    idxes = [i for i, e in enumerate(env.envs) if not e.game_over()]
                    if idxes:
                        for i in idxes:
                            obs[i] = env.envs[i].reset()
                    else:
                        break
        scores.append(env.score())
        epslens.append(env.epslen())
    
    if record:
        if env.n_envs == 1:
            imgs = np.array(imgs, copy=False)
            if len(imgs.shape) == 5:
                imgs = imgs.transpose((1, 0, 2, 3, 4))
        else:
            imgs = np.stack(imgs, axis=1)
        return scores, epslens, imgs
    else:
        return scores, epslens, None
