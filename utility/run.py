from collections import deque
import numpy as np
from utility.display import pwc
from utility.timer import TBTimer


class Runner:
    def __init__(self, env, agent, step=0):
        self.env = env
        self.agent = agent
        self.obs = env.reset()
        self.step = step

        self._frame_skip = getattr(env, 'frame_skip', 1)
        self._frames_per_step = self.env.n_envs * self._frame_skip
        self._default_nsteps = env.max_episode_steps // self._frame_skip

    def run(self, *, step_fn=None, nsteps=None):
        """ run `nstep` agent steps, auto reset if an episodes is done """
        env = self.env
        obs = self.obs
        nsteps = nsteps or self._default_nsteps
        terms = {}
        for t in range(nsteps):
            action = self.agent(obs, deterministic=False)
            if isinstance(action, tuple):
                action, terms = action
            next_obs, reward, done, _ = env.step(action)
            self.step += self._frames_per_step
            if step_fn:
                step_fn(env, self.step, obs=obs, action=action, reward=reward,
                    discount=1-done, nth_obs=next_obs, **terms)
            if np.any(env.already_done()):
                if env.n_envs == 1:
                    obs = env.reset()
                else:
                    for i, e in enumerate(env.envs):
                        if e.already_done():
                            obs[i] = e.reset()
            else: 
                obs = next_obs
        self.obs = obs
        return self.step

    def run_traj(self, *, step_fn=None):
        env = self.env
        obs = env.reset()
        terms = {}
        while True:
            action = self.agent(obs, deterministic=False)
            if isinstance(action, tuple):
                action, terms = action
            next_obs, reward, done, _ = env.step(action)
            discount = 1-done
            self.step += np.sum(discount)
            if step_fn:
                if env.n_envs == 1:
                    step_fn(env, self.step, obs=obs, action=action, reward=reward,
                        discount=1-done, nth_obs=next_obs, **terms)
                else:
                    step_fn(env, self.step, obs=obs, action=action, reward=reward,
                        discount=1-done, nth_obs=next_obs, mask=env.mask(), **terms)
            if np.all(env.game_over()):
                break
            else:
                obs = next_obs
        return self.step

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
        for k in range(10000):
            if record:
                if name.startswith('dm'):
                    imgs.append(obs)
                else:
                    imgs.append(env.get_screen(size=size))
            action = agent(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            
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
            if len(imgs.shape) == 5:
                imgs = imgs.transpose((1, 0, 2, 3, 4))
        else:
            imgs = np.stack(imgs, axis=1)
        return scores, epslens, imgs
    else:
        return scores, epslens, None
