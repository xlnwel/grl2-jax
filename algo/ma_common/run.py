from functools import partial
import collections
import numpy as np

from tools.run import *
from tools.store import StateStore
from tools.utils import batch_dicts


State = collections.namedtuple('state', 'agent runner')


class Runner(RunnerWithState):
  def eval_with_video(
    self, 
    agent, 
    n_envs=None, 
    name=None, 
    **kwargs
  ):
    if name is None:
      return self._eval_with_video(agent, **kwargs)
    else:
      def constructor():
        env_config = self.env_config()
        if n_envs:
          env_config.n_envs = n_envs
        agent_states = agent.build_memory()
        runner_states = self.build_env(env_config)
        return State(agent_states, runner_states)
      set_fn = partial(set_states, agent=agent, runner=self)

      with StateStore(name, constructor, set_fn):
        stats = self._eval_with_video(agent, **kwargs)
      return stats

  def _eval_with_video(
    self, 
    agent, 
    n=None, 
    record_video=True, 
    size=(128, 128), 
    video_len=1000, 
    n_windows=4
  ):
    if n is None:
      n = self.env.n_envs
    n_done_eps = 0
    n_run_eps = self.env.n_envs
    scores = []
    epslens = []
    frames = [collections.deque(maxlen=video_len) 
      for _ in range(min(n_windows, self.env.n_envs))]
    stats_list = []

    prev_done = np.zeros(self.env.n_envs)
    self.env.manual_reset()
    env_output = self.env.reset()
    while n_done_eps < n:
      if record_video:
        f = self.env.get_screen(size=size)
        if self.env.env_type == 'Env':
          frames[0].append(f)
        else:
          for i in range(len(frames)):
            frames[i].append(f[i])

      action, stats = agent(env_output)

      env_output = self.env.step(action)
      stats_list.append(stats)

      done = self.env.game_over()
      done_env_ids = [i for i, (d, pd) in 
        enumerate(zip(done, prev_done)) if d and not pd]
      n_done_eps += len(done_env_ids)

      if done_env_ids:
        score = self.env.score(done_env_ids)
        epslen = self.env.epslen(done_env_ids)
        scores += score
        epslens += epslen
        if n_run_eps < n:
          reset_env_ids = done_env_ids[:n-n_run_eps]
          n_run_eps += len(reset_env_ids)
          eo = self.env.reset(reset_env_ids)
          for t, s in zip(env_output, eo):
            if isinstance(t, dict):
              for k in t.keys():
                for i, ri in enumerate(reset_env_ids):
                  t[k][ri] = s[k][i]
            else:
              for i, ri in enumerate(reset_env_ids):
                t[ri] = s[i]
      prev_done = done

    stats = batch_dicts(stats_list)
    if record_video:
      max_len = np.max([len(f) for f in frames])
      # padding to make all sequences of the same length
      for i, f in enumerate(frames):
        while len(f) < max_len:
          f.append(f[-1])
        frames[i] = np.array(f)
      frames = np.array(frames)
      return scores, epslens, stats, frames
    else:
      return scores, epslens, stats, None
