import collections
import logging
import numpy as np

from env.wrappers import get_wrapper_by_name

logger = logging.getLogger(__name__)

class Runner:
    def __init__(self, env, agent, step=0, nsteps=None):
        self.env = env
        if env.max_episode_steps == int(1e9):
            logger.info(f'Maximum episode steps is not specified'
                f'and is by default set to {self.env.max_episode_steps}')
            assert nsteps is not None
        self.agent = agent
        self.step = step
        self.env_output = self.env.output()
        self.episodes = np.zeros(env.n_envs)
        assert get_wrapper_by_name(self.env, 'EnvStats').auto_reset
        self.run = {
            'Env': self._run_env,
            'EnvVec': self._run_envvec,
        }[self.env.env_type]

        self._frame_skip = getattr(env, 'frame_skip', 1)
        self._frames_per_step = self.env.n_envs * self._frame_skip
        self._default_nsteps = nsteps or env.max_episode_steps // self._frame_skip

    def _run_env(self, *, action_selector=None, step_fn=None, nsteps=None):
        action_selector = action_selector or self.agent
        nsteps = nsteps or self._default_nsteps
        obs = self.env_output.obs
        reset = self.env_output.reset
        terms = {}

        for t in range(nsteps):
            action = action_selector(
                obs, 
                reset=reset, 
                deterministic=False,
                env_output=self.env_output)
            if isinstance(action, tuple):
                if len(action) == 2:
                    action, terms = action
                    frame_skip = self._frames_per_step
                elif len(action) == 3:
                    action, frame_skip, terms = action
            self.env_output = self.env.step(action, frame_skip=frame_skip)
            next_obs, reward, discount, reset = self.env_output

            self.step += frame_skip
            if step_fn:
                kwargs = dict(obs=obs, action=action, reward=reward,
                    discount=discount, next_obs=next_obs)
                # allow terms to overwrite the values in kwargs
                kwargs.update(terms)
                step_fn(self.env, self.step, reset, **kwargs)
            obs = next_obs
            # logging when env is reset 
            if reset:
                info = self.env.info()
                if 'score' in info:
                    self.agent.store(
                        score=info['score'], epslen=info['epslen'])
                    self.episodes += 1

        return self.step

    def _run_envvec(self, *, action_selector=None, step_fn=None, nsteps=None):
        action_selector = action_selector or self.agent
        nsteps = nsteps or self._default_nsteps
        obs = self.env_output.obs
        reset = self.env_output.reset
        terms = {}

        for t in range(nsteps):
            action = action_selector(
                obs, 
                reset=reset, 
                deterministic=False,
                env_output=self.env_output)
            if isinstance(action, tuple):
                action, terms = action
            self.env_output = self.env.step(action)
            next_obs, reward, discount, reset = self.env_output
            
            self.step += self._frames_per_step
            if step_fn:
                kwargs = dict(obs=obs, action=action, reward=reward,
                    discount=discount, next_obs=next_obs)
                # allow terms to overwrite the values in kwargs
                kwargs.update(terms)
                step_fn(self.env, self.step, reset, **kwargs)
            obs = next_obs
            # logging when any env is reset 
            done_env_ids = [i for i, r in enumerate(reset) if r]
            if done_env_ids:
                info = self.env.info(done_env_ids)
                # further filter done caused by life loss
                done_env_ids = [k for k, i in enumerate(info) if i.get('game_over')]
                info = [info[i] for i in done_env_ids]
                score = [i['score'] for i in info]
                epslen = [i['epslen'] for i in info]
                self.agent.store(score=score, epslen=epslen)
                self.episodes[done_env_ids] += 1

        return self.step

    def run_traj(self, *, action_selector=None, step_fn=None):
        if self.env.env_type == 'Env':
            return self._run_traj_env(action_selector, step_fn)
        else:
            return self._run_traj_envvec(action_selector, step_fn)

    # def _run_traj_env(self, action_selector, step_fn):
    #     env = self.env
    #     action_selector = action_selector or self.agent
    #     env_output = env.reset()
    #     obs = env_output.obs
    #     reset = 1
    #     assert env.epslen() == 0
    #     terms = {}

    #     for t in range(self._default_nsteps):
    #         action = action_selector(
    #             obs, 
    #             reset=reset, 
    #             deterministic=False, 
    #             env_output=env_output)
    #         if isinstance(action, tuple):
    #             action, terms = action
    #         env_output = env.step(action)
    #         next_obs, reward, discount, reset = env_output

    #         self.step += info.get('mask', 1) * self._frames_per_step
    #         if step_fn:
    #             kwargs = dict(obs=obs, action=action, reward=reward,
    #                 discount=np.float32(1-done), next_obs=next_obs)
    #             # allow terms to overwrite the values in kwargs
    #             kwargs.update(terms)
    #             step_fn(env, self.step, info, **kwargs)
    #         obs = next_obs
    #         # logging and reset
    #         if info.get('already_done'):
    #             if info.get('game_over'):
    #                 self.agent.store(score=info['score'], epslen=info['epslen'])
    #                 self.episodes += 1
    #                 break
    #             else:
    #                 obs = env.reset()
    #             reset = 1

    #     return self.step

    # def _run_traj_envvec(self, action_selector, step_fn):
    #     env = self.env
    #     action_selector = action_selector or self.agent
    #     env_output = env.reset()
    #     obs = env_output.obs
    #     reset = np.ones(env.n_envs)
    #     np.testing.assert_equal(env.epslen(), np.zeros_like(reset))
    #     terms = {}

    #     for t in range(self._default_nsteps):
    #         action = action_selector(
    #             obs, 
    #             reset=reset, 
    #             deterministic=False, 
    #             env_output=env_output)
    #         if isinstance(action, tuple):
    #             action, terms = action
    #         env_output = env.step(action)
    #         next_obs, reward, done, info = env_output

    #         mask = np.array([i.get('mask', 1) for i in info])
    #         self.step += np.sum(self._frames_per_step * mask)
    #         if step_fn:
    #             kwargs = dict(obs=obs, action=action, reward=reward,
    #                 discount=np.float32(1-done), next_obs=next_obs, mask=mask)
    #             # allow terms to overwrite the values in kwargs
    #             kwargs.update(terms)
    #             step_fn(env, self.step, info, **kwargs)
    #         obs = next_obs
    #         # logging and reset 
    #         done_env_ids = [i for i, ii in enumerate(info) if ii.get('already_done')]
    #         if done_env_ids:
    #             score = [i['score'] for i in info if 'score' in i]
    #             if score:
    #                 epslen = [i['epslen'] for i in info if 'epslen' in i]
    #                 assert len(score) == len(epslen)
    #                 self.agent.store(score=score, epslen=epslen)
                
    #             reset_env_ids = [i for i in done_env_ids if not info[i].get('game_over')]
    #             if reset_env_ids:
    #                 new_obs = env.reset(reset_env_ids)
    #                 for i, o in zip(done_env_ids, new_obs):
    #                     obs[i] = o
    #         reset = np.array([i.get('already_done', False) for i in info])
    #         if np.all([i.get('game_over', False) for i in info]):
    #             break

    #     return self.step

def evaluate(env, agent, n=1, record=False, size=None, video_len=1000, step_fn=None):
    assert get_wrapper_by_name(env, 'EnvStats') is not None
    scores = []
    epslens = []
    max_steps = env.max_episode_steps // getattr(env, 'frame_skip', 1)
    maxlen = min(video_len, max_steps)
    frames = collections.defaultdict(lambda:collections.deque(maxlen=maxlen))
    name = env.name
    if hasattr(agent, 'reset_states'):
        agent.reset_states()
    env_output = env.reset()
    obs = env_output.obs
    n_run_eps = env.n_envs  # count the number of episodes that has begun to run
    n = max(n, env.n_envs)
    n_done_eps = 0
    while n_done_eps < n:
        for k in range(max_steps):
            if record:
                img = env.get_screen(size=size)
                if env.env_type == 'Env':
                    frames[0].append(img)
                else:
                    for i in range(env.n_envs):
                        frames[i].append(img[i])
                    
            action = agent(obs, deterministic=True, env_output=env_output)
            env_output = env.step(action)
            obs, reward, discount, reset = env_output

            if step_fn:
                step_fn(reward=reward)
            if env.env_type == 'Env':
                if env.game_over():
                    scores.append(env.score())
                    epslens.append(env.epslen())
                    n_done_eps += 1
                    if n_run_eps < n:
                        n_run_eps += 1
                        env_output = env.reset()
                        obs = env_output.obs
                        if hasattr(agent, 'reset_states'):
                            agent.reset_states()
                    break
            else:
                done_env_ids = [i for i, d in enumerate(env.game_over()) if d]
                n_done_eps += len(done_env_ids)
                if done_env_ids:
                    score = env.score(done_env_ids)
                    epslen = env.epslen(done_env_ids)
                    scores += score
                    epslens += epslen
                    if n_run_eps < n:
                        reset_env_ids = done_env_ids[:n-n_run_eps]
                        n_run_eps += len(reset_env_ids)
                        eo = env.reset(reset_env_ids)
                        for t, s in zip(env_output, eo):
                            for si, ti in enumerate(done_env_ids):
                                t[ti] = s[si]
                    elif len(done_env_ids) == env.n_envs:
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
