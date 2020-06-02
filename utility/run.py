import collections
import numpy as np


class Runner:
    def __init__(self, env, agent, step=0, nsteps=None):
        self.env = env
        self.agent = agent
        self.step = step
        self.env_output = None

        self.is_env = self.env.env_type == 'Env'
        self.aws = hasattr(self.agent, '_state')

        self.run = {
            (True, True): self._run_env_aws,
            (True, False):self._run_env,
            (False, True) :self._run_envvec_aws,
            (False, False): self._run_envvec,
        }[(self.is_env, self.aws)]

        self._frame_skip = getattr(env, 'frame_skip', 1)
        self._frames_per_step = self.env.n_envs * self._frame_skip
        self._default_nsteps = nsteps or env.max_episode_steps // self._frame_skip

    def _run_env(self, *, action_selector=None, step_fn=None, nsteps=None):
        if self.env_output is None:
            self.env_output = self.env.reset()
        action_selector = action_selector or self.agent
        nsteps = nsteps or self._default_nsteps
        obs = self.env_output.obs
        terms = {}

        for t in range(nsteps):
            action = action_selector(obs, deterministic=False)
            if isinstance(action, tuple):
                action, terms = action
            env_output = self.env.step(action)
            self.env_output = self.env.step(action)
            next_obs, reward, done, info = self.env_output

            self.step += self._frames_per_step
            if step_fn:
                kwargs = dict(obs=obs, action=action, reward=reward,
                    discount=1-done, next_obs=next_obs)
                # allow terms to overwrite the values in kwargs
                kwargs.update(terms)
                step_fn(self.env, self.step, info, **kwargs)
            obs = next_obs
            # logging and reset 
            if info.get('already_done'):
                if info.get('game_over'):
                    self.agent.store(score=info['score'], epslen=info['epslen'])
                self.env_output = self.env.reset()
                obs = self.env_output.obs

        return self.step

    def _run_env_aws(self, *, action_selector=None, reset_fn=None, step_fn=None, nsteps=None):
        if self.env_output is None:
            self.env_output = self.env.reset()
            self.reset = 1
            if reset_fn:
                reset_fn(**self.env_output._asdict())
        action_selector = action_selector or self.agent
        nsteps = nsteps or self._default_nsteps
        obs = self.env_output.obs
        terms = {}

        for t in range(nsteps):
            action = action_selector(
                obs, 
                reset=self.reset, 
                deterministic=False,
                env_output=self.env_output)
            if isinstance(action, tuple):
                action, terms = action
            self.env_output = self.env.step(action)
            next_obs, reward, done, info = self.env_output

            self.step += self._frames_per_step
            if step_fn:
                kwargs = dict(obs=obs, action=action, reward=reward,
                    discount=1-done, next_obs=next_obs)
                # allow terms to overwrite the values in kwargs
                kwargs.update(terms)
                step_fn(self.env, self.step, info, **kwargs)
            obs = next_obs
            # logging and reset 
            if info.get('already_done'):
                if info.get('game_over'):
                    self.agent.store(score=info['score'], epslen=info['epslen'])
                self.env_output = self.env.reset()
                obs = self.env_output.obs
                self.reset = 1
                if reset_fn:
                    reset_fn(**self.env_output._asdict())
            else:
                self.reset = 0

        return self.step

    def _run_envvec(self, *, action_selector=None, step_fn=None, nsteps=None):
        if self.env_output is None:
            self.env_output = self.env.reset()
        action_selector = action_selector or self.agent
        nsteps = nsteps or self._default_nsteps
        obs = self.env_output.obs
        terms = {}

        for t in range(nsteps):
            action = action_selector(obs, deterministic=False)
            if isinstance(action, tuple):
                action, terms = action
            self.env_output = self.env.step(action)
            next_obs, reward, done, info = self.env_output

            self.step += self._frames_per_step
            if step_fn:
                kwargs = dict(obs=obs, action=action, reward=reward,
                    discount=1-done, next_obs=next_obs)
                # allow terms to overwrite the values in kwargs
                kwargs.update(terms)
                step_fn(self.env, self.step, info, **kwargs)
            obs = next_obs
            # logging and reset 
            done_env_ids = [i for i, ii in enumerate(info) if ii.get('already_done')]
            if done_env_ids:
                score = [i['score'] for i in info if 'score' in i]
                if score:
                    epslen = [i['epslen'] for i in info if 'epslen' in i]
                    assert len(score) == len(epslen)
                    self.agent.store(score=score, epslen=epslen)
                
                env_output = self.env.reset(done_env_ids)
                for i, eo in zip(done_env_ids, zip(*env_output)):
                    for k, v in enumerate(eo):
                        self.env_output[k][i] = v
                obs = self.env_output.obs

        return self.step

    def _run_envvec_aws(self, *, action_selector=None, step_fn=None, nsteps=None):
        if self.env_output is None:
            self.env_output = self.env.reset()
            self.reset = np.ones(env.n_envs)
        action_selector = action_selector or self.agent
        nsteps = nsteps or self._default_nsteps
        obs = self.env_output.obs
        terms = {}

        for t in range(nsteps):
            action = action_selector(
                obs, 
                reset=self.reset, 
                deterministic=False,
                env_output=self.env_output)
            if isinstance(action, tuple):
                action, terms = action
            self.env_output = self.env.step(action)
            next_obs, reward, done, info = self.env_output

            self.step += self._frames_per_step
            if step_fn:
                kwargs = dict(obs=obs, action=action, reward=reward,
                    discount=1-done, next_obs=next_obs)
                # allow terms to overwrite the values in kwargs
                kwargs.update(terms)
                step_fn(self.env, self.step, info, **kwargs)
            obs = next_obs
            # logging and reset 
            done_env_ids = [i for i, ii in enumerate(info) if ii.get('already_done')]
            if done_env_ids:
                score = [i['score'] for i in info if 'score' in i]
                if score:
                    epslen = [i['epslen'] for i in info if 'epslen' in i]
                    assert len(score) == len(epslen)
                    self.agent.store(score=score, epslen=epslen)
                
                env_output = self.env.reset(done_env_ids)
                for i, eo in zip(done_env_ids, zip(*env_output)):
                    for k, v in enumerate(eo):
                        self.env_output[k][i] = v
                obs = self.env_output.obs
            self.reset = np.array([i.get('already_done', 0) for i in info])

        return self.step

    def run_traj(self, *, action_selector=None, step_fn=None):
        if self.env.env_type == 'Env':
            return self._run_traj_env(action_selector, step_fn)
        else:
            return self._run_traj_envvec(action_selector, step_fn)

    def _run_traj_env(self, action_selector, step_fn):
        env = self.env
        action_selector = action_selector or self.agent
        env_output = env.reset()
        obs = env_output.obs
        reset = 1
        assert env.epslen() == 0
        terms = {}

        for t in range(self._default_nsteps):
            action = action_selector(
                obs, 
                reset=reset, 
                deterministic=False, 
                env_output=env_output)
            if isinstance(action, tuple):
                action, terms = action
            env_output = env.step(action)
            next_obs, reward, done, info = env_output

            self.step += info.get('mask', 1) * self._frames_per_step
            if step_fn:
                kwargs = dict(obs=obs, action=action, reward=reward,
                    discount=1-done, next_obs=next_obs)
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
        env_output = env.reset()
        obs = env_output.obs
        reset = np.ones(env.n_envs)
        np.testing.assert_equal(env.epslen(), np.zeros_like(reset))
        terms = {}

        for t in range(self._default_nsteps):
            action = action_selector(
                obs, 
                reset=reset, 
                deterministic=False, 
                env_output=env_output)
            if isinstance(action, tuple):
                action, terms = action
            env_output = env.step(action)
            next_obs, reward, done, info = env_output

            mask = np.array([i.get('mask', 1) for i in info])
            self.step += np.sum(self._frames_per_step * mask)
            if step_fn:
                kwargs = dict(obs=obs, action=action, reward=reward,
                    discount=1-done, next_obs=next_obs, mask=mask)
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
                
                reset_env_ids = [i for i in done_env_ids if not info[i].get('game_over')]
                if reset_env_ids:
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
    frames = collections.defaultdict(lambda:collections.deque(maxlen=maxlen))
    name = env.name
    for _ in range(0, n, env.n_envs):
        if hasattr(agent, 'reset_states'):
            agent.reset_states()
        env_output = env.reset()
        obs = env_output.obs
        for k in range(env.max_episode_steps):
            if record:
                if name.startswith('dm'):
                    img = obs
                else:
                    img = env.get_screen(size=size)
                if env.n_envs == 1:
                    frames[0].append(img)
                for i in range(env.n_envs):
                    if not env_output.info[i].get('game_over'):
                        frames[i].append(img[i])
                    
            action = agent(obs, deterministic=True, env_output=env_output)
            env_output = env.step(action)
            obs, reward, done, info = env_output

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
                    if reset_env_ids:
                        new_obs = env.reset(reset_env_ids)
                        for i, o in zip(reset_env_ids, new_obs):
                            obs[i] = o
            if np.all(env.game_over()):
                break
    
    if record:
        frames = list(frames.values())
        max_len = np.max([len(f) for f in frames])
        # padding to make all sequences of the same length
        for i, f in enumerate(frames):
            while len(f) < max_len:
                f.append(f[-1])
            frames[i] = np.array(f)
        frames = np.array(frames)
        return scores, epslens, frames
    else:
        return scores, epslens, None
