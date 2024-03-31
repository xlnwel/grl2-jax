import collections
from typing import Tuple
from functools import partial
import logging
import numpy as np

from core.typing import dict2AttrDict
from tools.store import StateStore
from tools.utils import batch_dicts
from env.typing import EnvOutput
from env.func import create_env
from env.utils import divide_env_output

logger = logging.getLogger(__name__)

State = collections.namedtuple('state', 'agent runner')
EnvState = collections.namedtuple('state', 'env output')


class RunMode:
  NSTEPS='nsteps'
  TRAJ='traj'


def concat_along_unit_dim(x):
  x = np.concatenate(x, axis=1)
  return x


class RunnerWithState:
  def __init__(self, env_config, seed_interval=1000):
    self._env_config = dict2AttrDict(env_config, to_copy=True)
    self._seed_interval = seed_interval
    self.build_env(for_self=True)
  
  def __getattr__(self, name):
    if name.startswith("_"):
      raise AttributeError(
        "attempted to get missing private attribute '{}'".format(name)
      )
    return getattr(self.env, name)

  def env_config(self):
    return dict2AttrDict(self._env_config, to_copy=True)

  def env_stats(self):
    return self._env_stats

  def build_env(self, env_config=None, for_self=False):
    env_config = env_config or self._env_config
    env = create_env(env_config)
    env_output = env.output()
    if self._env_config.seed is not None:
      self._env_config.seed += self._seed_interval
    if for_self:
      self._env_stats = env.stats()
      self.env = env
      self.env_output = env_output
    return EnvState(env, env_output)

  def get_states(self):
    return EnvState(self.env, self.env_output)
  
  def reset_states(self):
    env = self.env
    env_output = self.env_output
    self.env = create_env(self._env_config)
    self.env_output = self.env.output()
    self._env_config.seed += self._seed_interval
    return EnvState(env, env_output)

  def set_states(self, state):
    curr_state = EnvState(self.env, self.env_output)
    self.env, self.env_output = state
    return curr_state

  def run(
    self, 
    agents: Tuple, 
    *, 
    name=None, 
    env_kwargs={}, 
    **kwargs, 
  ):
    if name is None:
      return self._run(agents, **kwargs)
    else:
      constructor = partial(
        state_constructor, agents=agents, runner=self, env_kwargs=env_kwargs)
      set_fn = partial(set_states, agents=agents, runner=self)
      with StateStore(name, constructor, set_fn):
        return self._run(agents, **kwargs)

  def _run(self, agents, **kwargs):
    self.env_output = run(
      agents, 
      self.env, 
      self.env_output, 
      self._env_stats, 
      **kwargs
    )
    return self.env_output

  def get_steps_per_run(self, n_steps):
    return self._env_stats.n_envs * n_steps


def run(
  agents: Tuple, 
  env, 
  env_output, 
  env_stats, 
  n_steps, 
  store_info=True, 
  collect_data=True, 
  eps_callbacks=[]
):
  for _ in range(n_steps):
    agent_env_outputs = divide_env_output(env_output)
    actions, stats = zip(*[a(o) for a, o in zip(agents, agent_env_outputs)])
    new_env_output = env.step(actions)
    new_agent_env_outputs = divide_env_output(new_env_output)
    assert len(agent_env_outputs) == len(actions) == len(new_agent_env_outputs)

    if collect_data:
      next_obs = env.prev_obs()
      for i, agent in enumerate(agents):
        data = dict(
          obs=env_output[i].obs, 
          action=actions[i], 
          reward=new_agent_env_outputs[i].reward, 
          discount=new_agent_env_outputs[i].discount, 
          next_obs=next_obs[i], 
          reset=new_agent_env_outputs[i].reset,
        )
        data.update(stats[i])
        agent.buffer.collect(**data)

    env_output = new_env_output
    done_env_ids = [i for i, r in enumerate(env_output.reset[0]) if np.all(r)]
    if done_env_ids:
      info = env.info(done_env_ids, convert_batch=True)
      if store_info:
        for aid, uids in enumerate(env_stats.aid2uids):
          agent_info = {k: [vv[uids] for vv in v]
              if isinstance(v[0], np.ndarray) else v 
              for k, v in info.items()}
          agents[aid].store(**agent_info)

      for callback in eps_callbacks:
        callback(info=info)

  return env_output


class Runner:
  def __init__(self, env, agent, step=0, nsteps=None, 
        run_mode=RunMode.NSTEPS, record_envs=None, info_func=None):
    self.env = env
    if env.max_episode_steps == int(1e9):
      logger.info(f'Maximum episode steps is not specified'
        f'and is by default set to {self.env.max_episode_steps}')
      # assert nsteps is not None
    self.agent = agent
    self.step = step
    if run_mode == RunMode.TRAJ and env.env_type == 'VecEnv':
      logger.warning('Runner.step is not the actual environment steps '
        f'as run_mode == {RunMode.TRAJ} and env_type == VecEnv')
    self.env_output = self.env.output()
    self.episodes = np.zeros(env.n_envs)
    assert getattr(self.env, 'auto_reset', None), getattr(self.env, 'auto_reset', None)
    self.run = {
      f'{RunMode.NSTEPS}-Env': self._run_env,
      f'{RunMode.NSTEPS}-VecEnv': self._run_envvec,
      f'{RunMode.TRAJ}-Env': self._run_traj_env,
      f'{RunMode.TRAJ}-VecEnv': self._run_traj_envvec,
    }[f'{run_mode}-{self.env.env_type}']

    self._frame_skip = getattr(env, 'frame_skip', 1)
    self._frames_per_step = self.env.n_envs * self._frame_skip
    self._default_nsteps = nsteps or env.max_episode_steps // self._frame_skip
    self._is_multi_agent = self.env.stats().is_multi_agent

    record_envs = record_envs or self.env.n_envs
    self._record_envs = list(range(record_envs))

    self._info_func = info_func

  def reset(self):
    self.env_output = self.env.reset()

  def _run_env(self, *, action_selector=None, step_fn=None, nsteps=None):
    action_selector = action_selector or self.agent
    nsteps = nsteps or self._default_nsteps
    obs = self.env_output.obs

    for t in range(nsteps):
      action = action_selector(self.env_output, evaluation=False)
      obs, reset = self.step_env(obs, action, step_fn)

      # logging when env is reset 
      if reset:
        info = self.env.info()
        if 'score' in info:
          self.store_info(info)
          self.episodes += 1

    return self.step

  def _run_envvec(self, *, action_selector=None, step_fn=None, nsteps=None):
    action_selector = action_selector or self.agent
    nsteps = nsteps or self._default_nsteps
    obs = self.env_output.obs
    
    for t in range(nsteps):
      action = action_selector(self.env_output, evaluation=False)
      obs, reset = self.step_env(obs, action, step_fn)

      # logging when any env is reset 
      if self._is_multi_agent:
        reset = reset[0]

      done_env_ids = [i for i, r in enumerate(reset)
        if (np.all(r) if isinstance(r, np.ndarray) else r) 
      and i in self._record_envs]

      if done_env_ids:
        info = self.env.info(done_env_ids)
        if info:
          self.store_info(info)
        self.episodes[done_env_ids] += 1

    return self.step

  def _run_traj_env(self, action_selector=None, step_fn=None):
    action_selector = action_selector or self.agent
    obs = self.env_output.obs
    
    for t in range(self._default_nsteps):
      action = action_selector(self.env_output, evaluation=False)
      obs, reset = self.step_env(obs, action, step_fn)

      if reset:
        break
    
    info = self.env.info()
    self.store_info(info)
    self.episodes += 1

    return self.step

  def _run_traj_envvec(self, action_selector=None, step_fn=None):
    action_selector = action_selector or self.agent
    obs = self.env_output.obs
    
    for t in range(self._default_nsteps):
      action = action_selector(self.env_output, evaluation=False)
      obs, reset = self.step_env(obs, action, step_fn)

      # logging when any env is reset 
      if np.all(reset):
        break

    info = [i for idx, i in enumerate(self.env.info()) if idx in self._record_envs]
    self.store_info(info)
    self.episodes += 1

    return self.step

  def step_env(self, obs, action, step_fn):
    if isinstance(action, tuple):
      if len(action) == 2:
        action, stats = action
        self.env_output = self.env.step(action)
        self.step += self._frames_per_step
      elif len(action) == 3:
        action, frame_skip, stats = action
        frame_skip += 1   # plus 1 as values returned start from zero
        self.env_output = self.env.step(action, frame_skip=frame_skip)
        self.step += np.sum(frame_skip)
      else:
        raise ValueError(f'Invalid action "{action}"')
    else:
      self.env_output = self.env.step(action)
      self.step += self._frames_per_step
      stats = {}

    next_obs, reward, discount, reset = self.env_output

    if step_fn:
      kwargs = dict(
        obs=obs[0] if self._is_multi_agent else obs, 
        action=action[0] if self._is_multi_agent and isinstance(action, list) else action, 
        reward=reward[0] if self._is_multi_agent else reward,
        discount=discount[0] if self._is_multi_agent else discount, 
        next_obs=next_obs[0] if self._is_multi_agent else next_obs
      )
      assert 'reward' not in stats, 'reward in stats is from the preivous timestep and should not be used to override here'
      # allow stats to overwrite the values in kwargs
      kwargs.update(stats)
      step_fn(self.env, self.step, reset[0] if self._is_multi_agent else reset, **kwargs)

    return next_obs, reset
  
  def store_info(self, info):
    info = batch_dicts(info, list)
    self.agent.store(**info)
    if self._info_func is not None:
      self._info_func(self.agent, info)


def simple_evaluate(env, agents, n_eps, render=False):
  assert env.n_envs == 1, env.n_envs
  agent_scores = [[] for _ in agents]
  epslen = []
  eps_i = 0

  env_output = env.output()
  while eps_i < n_eps:
    agent_outputs = divide_env_output(env_output)
    assert len(agent_outputs) == len(agents), (len(agent_outputs), len(agents))
    actions = [a(o, evaluation=True)[0] for a, o in zip(agents, agent_outputs)]
    env_output = env.step(actions)
    if render:
      env.render()
    if np.all(env_output.reset):
      agent_scores = [a + s for a, s in zip(agent_scores, zip(*env.score()))]
      epslen += env.epslen()
      eps_i += 1
  
  return agent_scores, epslen


def evaluate(
  env, 
  agents, 
  n=1, 
  record_video=False, 
  size=None, 
  video_len=1000, 
  n_windows=4
):
  scores = [[] for _ in agents]
  epslens = [[] for _ in agents]
  stats_list = [[] for _ in agents]
  if size is not None and len(size) == 1:
    size = size * 2
  max_steps = env.max_episode_steps // getattr(env, 'frame_skip', 1)
  frames = [collections.deque(maxlen=video_len) 
    for _ in range(min(n_windows, env.n_envs))]
  for a in agents:
    a.reset_states()
  n_run_eps = env.n_envs  # count the number of episodes that has begun to run
  n = max(n, env.n_envs)
  n_done_eps = 0
  prev_done = np.zeros(env.n_envs)
  env_output = env.output()
  while n_done_eps < n:
    for k in range(max_steps):
      if record_video:
        img = env.get_screen(size=size)
        if env.env_type == 'Env':
          frames[0].append(img)
        else:
          for i in range(len(frames)):
            frames[i].append(img[i])

      outs = divide_env_output(env_output)
      action, stats = zip(*[a(o, evaluation=True) for a, o in zip(agents, outs)])
      env_output = env.step(action)
      for sl, s, inf in zip(stats_list, stats, env.info()):
        s.update(inf)
        sl.append(s)

      done = env.game_over()
      done_env_ids = [i for i, (d, pd) in 
        enumerate(zip(done, prev_done)) if d and not pd]
      n_done_eps += len(done_env_ids)
      if done_env_ids:
        agent_score = list(zip(*env.score(done_env_ids)))
        agent_epslen = list(zip(*env.epslen(done_env_ids)))
        for i, _ in enumerate(agents):
          scores[i] += list(agent_score[i])
          epslens[i] += list(agent_epslen[i])
        if n_run_eps < n:
          reset_env_ids = done_env_ids[:n-n_run_eps]
          n_run_eps += len(reset_env_ids)
          eo = env.reset(reset_env_ids)
          for t, s in zip(env_output, eo):
            if isinstance(t, dict):
              for k in t.keys():
                for i, ri in enumerate(reset_env_ids):
                  t[k][ri] = s[k][i]
            else:
              for i, ri in enumerate(reset_env_ids):
                t[ri] = s[i]
        elif n_done_eps == n:
          break
      prev_done = done

  stats = [batch_dicts(s) for s in stats_list]
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


def state_constructor(agents, runner: RunnerWithState, env_kwargs={}):
  env_config = runner.env_config()
  env_config.update(env_kwargs)
  agent_states = []
  for a in agents:
    agent_states.append(a.build_memory())
  runner_states = runner.build_env(env_config)
  return State(agent_states, runner_states)


def set_states(states: State, agents: Tuple, runner):
  agent_states, runner_states = states
  for a, s in zip(agents, agent_states):
    a.set_memory(s)
  runner_states = runner.set_states(runner_states)
  return State(agent_states, runner_states)
