import collections
from functools import partial
import logging
import numpy as np

from core.typing import dict2AttrDict
from tools.store import StateStore
from tools.utils import batch_dicts, prefix_name
from env.func import create_env

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
    agent, 
    *, 
    name=None, 
    env_kwargs={}, 
    **kwargs, 
  ):
    if name is None:
      return self._run(agent, **kwargs)
    else:
      constructor = partial(
        state_constructor, agent=agent, runner=self, env_kwargs=env_kwargs)
      set_fn = partial(set_states, agent=agent, runner=self)
      with StateStore(name, constructor, set_fn):
        return self._run(agent, **kwargs)
  
  def _run(
    self, 
    agent, 
    n_steps, 
    store_info=True, 
    collect_data=True, 
  ):
    env_output = self.env_output
    for _ in range(n_steps):
      action, stats = agent(env_output)
      new_env_output = self.env.step(action)

      if collect_data:
        data = dict(
          obs=batch_dicts(env_output.obs, func=concat_along_unit_dim), 
          action=action, 
          reward=concat_along_unit_dim(new_env_output.reward), 
          discount=concat_along_unit_dim(new_env_output.discount), 
          next_obs=batch_dicts(self.env.prev_obs(), func=concat_along_unit_dim), 
          reset=concat_along_unit_dim(new_env_output.reset),
        )
        agent.buffer.collect(**data, **stats)

      if store_info:
        done_env_ids = [i for i, r in enumerate(new_env_output.reset[0]) if np.all(r)]

        if done_env_ids:
          info = self.env.info(done_env_ids)
          if info:
            info = batch_dicts(info, list)
            agent.store(**info)
      env_output = new_env_output

    self.env_output = env_output

    return env_output

  def get_steps_per_run(self, n_steps):
    return self._env_stats.n_envs * n_steps
  

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


def evaluate(
  env, 
  agent, 
  n=1, 
  record_video=False, 
  size=None, 
  video_len=1000, 
  step_fn=None, 
  n_windows=4
):
  scores = []
  epslens = []
  stats_list = []
  if size is not None and len(size) == 1:
    size = size * 2
  max_steps = env.max_episode_steps // getattr(env, 'frame_skip', 1)
  frames = [collections.deque(maxlen=video_len) 
    for _ in range(min(n_windows, env.n_envs))]
  if hasattr(agent, 'reset_states'):
    agent.reset_states()
  env.manual_reset()
  env_output = env.reset()
  n_run_eps = env.n_envs  # count the number of episodes that has begun to run
  n = max(n, env.n_envs)
  n_done_eps = 0
  obs = env_output.obs
  prev_done = np.zeros(env.n_envs)
  while n_done_eps < n:
    for k in range(max_steps):
      if record_video:
        img = env.get_screen(size=size)
        if env.env_type == 'Env':
          frames[0].append(img)
        else:
          for i in range(len(frames)):
            frames[i].append(img[i])

      action, stats = agent(
        env_output, 
        evaluation=True, 
      )
      stats_list.append(stats)
      env_output = env.step(action)
      next_obs, reward, discount, reset = env_output
      stats['reward'] = np.squeeze(reward)
      info = env.info()
      if isinstance(info, list):
        for i in info:
          stats.update(i)
      else:
        stats.update(info)

      if step_fn:
        step_fn(obs=obs, action=action, reward=reward, 
          discount=discount, next_obs=next_obs, 
          reset=reset, **stats)
      obs = next_obs
      if env.env_type == 'Env':
        if env.game_over():
          scores.append(env.score())
          epslens.append(env.epslen())
          n_done_eps += 1
          if n_run_eps < n:
            n_run_eps += 1
            env_output = env.reset()
            if hasattr(agent, 'reset_states'):
              agent.reset_states()
          break
      else:
        done = env.game_over()
        done_env_ids = [i for i, (d, pd) in 
          enumerate(zip(done, prev_done)) if d and not pd]
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


def state_constructor(agent, runner: RunnerWithState, env_kwargs={}):
  env_config = runner.env_config()
  env_config.update(env_kwargs)
  agent_states = agent.build_memory()
  runner_states = runner.build_env(env_config)
  return State(agent_states, runner_states)


def set_states(states: State, agent, runner):
  agent_states, runner_states = states
  agent_states = agent.set_memory(agent_states)
  runner_states = runner.set_states(runner_states)
  return State(agent_states, runner_states)
