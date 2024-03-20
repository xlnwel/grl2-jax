import numpy as np
import ray

from core.typing import AttrDict2dict
from env.cls import *
from env.typing import EnvOutput
from env.utils import batch_env_output
from tools.utils import convert_batch_with_func, batch_dicts


class RayVecEnv:
  def __init__(self, EnvType, config, env_fn=make_env):
    self.env_type = 'VecEnv'
    self.name = config['env_name']
    self.n_runners= config.get('n_runners', 1)
    self.envsperworker = config.get('n_envs', 1)
    self.n_envs = self.envsperworker * self.n_runners
    RayEnvType = ray.remote(EnvType)
    # leave the name "envs" for consistency, albeit workers seems more appropriate
    self.envs = []
    config = AttrDict2dict(config)
    for i in range(self.n_runners):
      config['seed'] = config.get('seed', 0) + i * self.envsperworker
      config['eid'] = config.get('eid', 0) + i * self.envsperworker
      self.envs.append(RayEnvType.remote(config, env_fn))

    self.env = EnvType(config, env_fn)
    self.max_episode_steps = self.env.max_episode_steps
    self._combine_func = np.stack

    self._stats = self.env.stats()
    self._stats['n_runners'] = self.n_runners
    self._stats['n_envs'] = self.n_envs

  def __getattr__(self, name):
    if name.startswith("_"):
      raise AttributeError(
        "attempted to get missing private attribute '{}'".format(name)
      )
    return getattr(self.env, name)

  def reset(self, idxes=None):
    out = self._remote_call('reset', idxes, single_output=False)
    return EnvOutput(*out)

  def random_action(self, *args, **kwargs):
    action = ray.get([env.random_action.remote() for env in self.envs])
    action = [batch_dicts(a, np.concatenate) for a in zip(*action)]
    return action

  def step(self, actions, **kwargs):
    if not isinstance(actions, (list, tuple)):
      actions = [actions]
    new_actions = []
    for i in range(self.n_runners):
      new_actions.append([{
        k: v[i*self.envsperworker: (i+1)*self.envsperworker] for k, v in action.items()
      } for action in actions])
    actions = new_actions
    if kwargs:
      kwargs = {k: [np.squeeze(x) for x in np.split(v, self.n_runners)] 
        for k, v in kwargs.items()}
      kwargs = [dict(x) for x in zip(*[itertools.product([k], v) 
        for k, v in kwargs.items()])]
      out = ray.get([env.step.remote(a, convert_batch=False, **kw) 
        for env, a, kw in zip(self.envs, actions, kwargs)])
    else:
      out = ray.get([env.step.remote(a, convert_batch=False) 
        for env, a in zip(self.envs, actions)])

    out = self._process_list_outputs(out, convert_batch=True)
    out = EnvOutput(*out)
    return out

  def score(self, idxes=None):
    return self._remote_call('score', idxes, convert_batch=False)

  def epslen(self, idxes=None):
    return self._remote_call('epslen', idxes, convert_batch=False)

  def game_over(self, idxes=None):
    return self._remote_call('game_over', idxes)

  def prev_obs(self, idxes=None):
    return self._remote_call('prev_obs', idxes)

  def info(self, idxes=None, convert_batch=False):
    return self._remote_call(
      'info', idxes, convert_batch=convert_batch)
  
  def output(self, idxes=None):
    out = self._remote_call('output', idxes, single_output=False)
    return out

  def _remote_call(self, name, idxes, single_output=True, convert_batch=True):
    """
    single_output: if the call produces only one output
    """
    method = lambda e: getattr(e, name)
    if idxes is None:
      out = ray.get([method(e).remote(convert_batch=False) for e in self.envs])
    else:
      if isinstance(self.env, Env):
        out = ray.get([
          method(self.envs[i]).remote(convert_batch=False) for i in idxes])
      else:
        new_idxes = [[] for _ in range(self.n_runners)]
        for i in idxes:
          new_idxes[i // self.envsperworker].append(i % self.envsperworker)
        out = ray.get([method(self.envs[i]).remote(j, convert_batch=False) 
          for i, j in enumerate(new_idxes) if j])

    if single_output:
      out = self._process_output(out, convert_batch=convert_batch)
      return out
    else:
      out = self._process_list_outputs(out, convert_batch=convert_batch)
      return out

  def _process_output(self, out, convert_batch):
    out = self._get_vectorized_outputs(out)
    if convert_batch:
      if isinstance(out[0], (list, tuple)):
        out = [convert_batch_with_func(o, func=self._combine_func) for o in zip(*out)]
      elif isinstance(out[0], dict):
        out = batch_dicts(out, self._combine_func)
    return out

  def _process_list_outputs(self, out, convert_batch):
    out = self._get_vectorized_outputs(out)
    if convert_batch:
      out = batch_env_output(out, func=self._combine_func)
    else:
      out = list(zip(*out))

    return out

  def _get_vectorized_outputs(self, out):
    if not isinstance(self.env, Env):
      # for these outputs, we expect them to be of form [[out*], [out*]]
      # and we chain them into [out*]
      out = list(itertools.chain(*out))
    return out

  def seed(self, seed=None):
    ray.get([e.seed(seed) for e in self.envs])

  def close(self):
    ray.get([env.close.remote() for env in self.envs])
    self.env.close()
    del self
