from collections import deque
import numpy as np


class Runner:
    def __init__(self, env, agent, step=0, nsteps=None):
        self.env = env
        self.agent = agent
        self.obs = env.reset()
        self.reset = np.ones(env.n_envs)
        self.step = step

        self._frame_skip = getattr(env, 'frame_skip', 1)
        self._frames_per_step = self.env.n_envs * self._frame_skip
        self._default_nsteps = nsteps or env.max_episode_steps // self._frame_skip

    def run(self, *, action_selector=None, step_fn=None, nsteps=None):
        """ run `nstep` agent steps, auto reset if an episodes is done """
        if self.env.env_type == 'Env':
            return self._run_env(action_selector, step_fn, nsteps)
        else:
            return self._run_envvec(action_selector, step_fn, nsteps)

    def _run_env(self, action_selector, step_fn, nsteps):
        env = self.env
        obs = self.obs
        action_selector = action_selector or self.agent
        nsteps = nsteps or self._default_nsteps
        terms = {}

        for t in range(nsteps):
            action = action_selector(obs, 
                        reset=self.reset, deterministic=False)
            if isinstance(action, tuple):
                action, terms = action
            next_obs, reward, done, info = env.step(action)

            self.step += self._frames_per_step
            if step_fn:
                kwargs = dict(obs=obs, action=action, reward=reward,
                    discount=1-done, nth_obs=next_obs)
                # allow terms to overwrite the values in kwargs
                kwargs.update(terms)
                step_fn(env, self.step, **kwargs)
            obs = next_obs
            # logging and reset 
            if info.get('already_done'):
                if info.get('game_over'):
                    self.agent.store(score=info['score'], epslen=info['epslen'])
                obs = env.reset()
            self.reset = env.already_done()
        self.obs = obs

        return self.step

    def _run_envvec(self, action_selector, step_fn, nsteps):
        env = self.env
        obs = self.obs
        action_selector = action_selector or self.agent
        nsteps = nsteps or self._default_nsteps
        terms = {}

        for t in range(nsteps):
            action = action_selector(obs, 
                        reset=self.reset, deterministic=False)
            if isinstance(action, tuple):
                action, terms = action
            next_obs, reward, done, info = env.step(action)

            self.step += self._frames_per_step
            if step_fn:
                kwargs = dict(obs=obs, action=action, reward=reward,
                    discount=1-done, nth_obs=next_obs)
                # allow terms to overwrite the values in kwargs
                kwargs.update(terms)
                step_fn(env, self.step, **kwargs)
            obs = next_obs
            # logging and reset 
            done_env_ids = [i for i, ii in enumerate(info) if ii.get('already_done')]
            if done_env_ids:
                score = [i['score'] for i in info if 'score' in i]
                if score:
                    epslen = [i['epslen'] for i in info if 'epslen' in i]
                    assert len(score) == len(epslen)
                    self.agent.store(score=score, epslen=epslen)
                
                new_obs = env.reset(done_env_ids)
                for i, o in zip(done_env_ids, new_obs):
                    obs[i] = o
            self.reset = np.array([i.get('already_done', False) for i in info])
        self.obs = obs

        return self.step

    def run_traj(self, *, action_selector=None, step_fn=None):
        if self.env.env_type == 'Env':
            return self._run_traj_env(action_selector, step_fn)
        else:
            return self._run_traj_envvec(action_selector, step_fn)

    def _run_traj_env(self, action_selector, step_fn):
        env = self.env
        action_selector = action_selector or self.agent
        obs = env.reset()
        reset = 1
        assert env.epslen() == 0
        terms = {}

        for t in range(self._default_nsteps):
            action = action_selector(obs, 
                        reset=reset, deterministic=False)
            if isinstance(action, tuple):
                action, terms = action
            next_obs, reward, done, info = env.step(action)

            self.step += info.get('mask', 1) * self._frames_per_step
            if step_fn:
                kwargs = dict(obs=obs, action=action, reward=reward,
                    discount=1-done, nth_obs=next_obs)
                # allow terms to overwrite the values in kwargs
                kwargs.update(terms)
                step_fn(env, self.step, **kwargs)
            obs = next_obs
            # logging and reset
            if info.get('already_done'):
                if info.get('game_over'):
                    self.agent.store(score=info['score'], epslen=info['epslen'])
                    break
                else:
                    obs = env.reset()
            reset = env.already_done()

        return self.step

    def _run_traj_envvec(self, action_selector, step_fn):
        env = self.env
        action_selector = action_selector or self.agent
        obs = env.reset()
        reset = np.ones(env.n_envs)
        np.testing.assert_equal(env.epslen(), np.zeros_like(reset))
        terms = {}

        for t in range(self._default_nsteps):
            action = action_selector(obs, 
                        reset=reset, deterministic=False)
            if isinstance(action, tuple):
                action, terms = action
            next_obs, reward, done, info = env.step(action)

            mask = np.array([i.get('mask', 1) for i in info])
            self.step += np.sum(self._frames_per_step * mask)
            if step_fn:
                kwargs = dict(obs=obs, action=action, reward=reward,
                    discount=1-done, nth_obs=next_obs)
                # allow terms to overwrite the values in kwargs
                kwargs.update(terms)
                kwargs['mask'] = mask
                step_fn(env, self.step, **kwargs)
            obs = next_obs
            # logging and reset 
            done_env_ids = [i for i, ii in enumerate(info) if ii.get('already_done')]
            if done_env_ids:
                score = [i['score'] for i in info if 'score' in i]
                if score:
                    epslen = [i['epslen'] for i in info if 'epslen' in i]
                    assert len(score) == len(epslen)
                    self.agent.store(score=score, epslen=epslen)
                
                reset_env_ids = [i for i in done_env_ids if not info[i].get('game_over')]
                new_obs = env.reset(reset_env_ids)
                for i, o in zip(done_env_ids, new_obs):
                    obs[i] = o
            reset = np.array([i.get('already_done', False) for i in info])
            if np.all([i.get('game_over', False) for i in info]):
                break

        return self.step

def evaluate(env, agent, n=1, record=False, size=None, video_len=1000):
    scores = []
    epslens = []
    maxlen = min(video_len, env.max_episode_steps)
    imgs = deque(maxlen=maxlen)
    name = env.name
    for _ in range(0, n, env.n_envs):
        if hasattr(agent, 'reset_states'):
            agent.reset_states()
        obs = env.reset()
        for k in range(env.max_episode_steps):
            if record:
                if name.startswith('dm'):
                    imgs.append(obs)
                else:
                    imgs.append(env.get_screen(size=size))
            action = agent(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            if env.n_envs == 1:
                if info.get('already_done'):
                    if info.get('game_over'):
                        scores.append(info['score'])
                        epslens.append(info['epslen'])
                    else:
                        obs = env.reset()
            else:
                done_env_ids = [i for i, ii in enumerate(info) if ii.get('already_done')]
                if done_env_ids:
                    score = [i['score'] for i in info if 'score' in i]
                    if score:
                        epslen = [i['epslen'] for i in info if 'epslen' in i]
                        scores += score
                        epslens += epslen
                    
                    reset_env_ids = [i for i in done_env_ids if not info[i].get('game_over')]
                    new_obs = env.reset(reset_env_ids)
                    for i, o in zip(reset_env_ids, new_obs):
                        obs[i] = o
            if np.all(env.game_over()):
                break
    
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
