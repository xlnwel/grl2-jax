import itertools
import random
import numpy as np
import cv2

from core.typing import dict2AttrDict
from tools.utils import batch_dicts, convert_batch_with_func
from env import make_env
from env.typing import EnvOutput
from env.utils import batch_env_output


class Env:
  def __init__(self, config, env_fn=make_env, agents={}):
    self.env = env_fn(config, eid=None, agents=agents)
    self.name = config['env_name']
    self.max_episode_steps = self.env.max_episode_steps
    self.n_envs = getattr(self.env, 'n_envs', 1)
    self.env_type = 'Env'
    self._stats = self.env.stats()
    self._stats['n_runners'] = 1
    self._stats['n_envs'] = self.n_envs
  
  def __getattr__(self, name):
    if name.startswith("_"):
      raise AttributeError(
        "attempted to get missing private attribute '{}'".format(name)
      )
    return getattr(self.env, name)

  def reset(self, idxes=None):
    return self.env.reset()

  def random_action(self, *args, **kwargs):
    action = self.env.random_action() if hasattr(self.env, 'random_action') \
      else self.env.action_space.sample()
    return action
    
  def step(self, action, *, convert_batch=False, **kwargs):
    output = self.env.step(action, **kwargs)
    return output

  def stats(self):
    return dict2AttrDict(self._stats)

  def manual_reset(self):
    self.env.manual_reset()

  """ the following code is required by ray """
  def score(self, *args):
    return self.env.score()

  def epslen(self, *args):
    return self.env.epslen()

  def mask(self, *args):
    return self.env.mask()

  def prev_obs(self, *args, **kwargs):
    return self.env.prev_obs()

  def info(self, *args, **kwargs):
    return self.env.info(*args)

  def prev_output(self, *args, **kwargs):
    return self.env.prev_output()
  
  def output(self, *args, **kwargs):
    return self.env.output()

  def game_over(self):
    return self.env.game_over()

  def get_screen(self, size=None):
    if hasattr(self.env, 'get_screen'):
      img = self.env.get_screen()
    else:
      img = self.env.render(mode='rgb_array')

    if size is not None and size != img.shape[:2]:
      # cv2 receive size of form (width, height)
      img = cv2.resize(img, size[::-1], interpolation=cv2.INTER_AREA)
      
    return img

  def record_default_state(self, aid, state):
    self.env.record_default_state(aid, state)

  def close(self):
    if hasattr(self.env, 'close'):
      self.env.close()


class VecEnvBase:
  def __init__(self, config, env_fn=make_env, agents={}):
    self.n_envs = n_envs = config.pop('n_envs', 1)
    self.name = config['env_name']
    if 'eid' not in config:
      config['eid'] = 0
    self.envs = []
    for i in range(n_envs):
      config_i = config.copy()
      config_i['eid'] += i
      config_i['seed'] += i
      env = env_fn(config_i, config_i['eid'])
      # env.seed(config_i['seed'])
      self.envs.append(env)
    self.env = self.envs[0]
    self.max_episode_steps = self.env.max_episode_steps
    self.env_type = 'VecEnv'

    self._stats = self.env.stats()
    self._stats['n_runners'] = 1
    self._stats['n_envs'] = self.n_envs

  def __getattr__(self, name):
    if name.startswith("_"):
      raise AttributeError(
        "attempted to get missing private attribute '{}'".format(name)
      )
    return getattr(self.env, name)

  def stats(self):
    return dict2AttrDict(self._stats)

  def _get_idxes(self, idxes):
    if idxes is None:
      idxes = list(range(self.n_envs))
    elif isinstance(idxes, int):
      idxes = [idxes]
    return idxes
  
  def process_output(self, out, convert_batch=True):
    if convert_batch:
      return batch_env_output(out)
    else:
      return out

  def combine_actions(self, actions):
    new_actions = [batch_dicts(a) for a in zip(*actions)]
    return new_actions

  def divide_actions(self, actions):
    new_actions = [
      [{k: v[i] for k, v in action.items()} for action in actions]
      for i in range(self.n_envs)
    ]
    return new_actions


class VecEnv(VecEnvBase):
  def __init__(self, config, env_fn=make_env, agents={}):
    super().__init__(config, env_fn, agents)

  def random_action(self, *args, **kwargs):
    actions = [env.random_action() for env in self.envs]
    return self.combine_actions(actions)

  def reset(self, idxes=None, convert_batch=True, **kwargs):
    idxes = self._get_idxes(idxes)
    out = [self.envs[i].reset() for i in idxes]
    out = self.process_output(out, convert_batch=convert_batch)
    return out

  def step(self, actions, convert_batch=True, **kwargs):
    actions = self.divide_actions(actions)
    assert len(self.envs) == len(actions), (len(self.envs), len(actions))
    outs = [e.step(a) for e, a in zip(self.envs, actions)]
    out = self.process_output(outs, convert_batch=convert_batch)
    return out

  def manual_reset(self):
    [e.manual_reset() for e in self.envs]

  def score(self, idxes=None, **kwargs):
    idxes = self._get_idxes(idxes)
    return [self.envs[i].score() for i in idxes]

  def epslen(self, idxes=None, **kwargs):
    idxes = self._get_idxes(idxes)
    return [self.envs[i].epslen() for i in idxes]

  def mask(self, idxes=None):
    idxes = self._get_idxes(idxes)
    return np.stack([self.envs[i].mask() for i in idxes])

  def game_over(self):
    return np.stack([env.game_over() for env in self.envs])

  def prev_obs(self, idxes=None, convert_batch=True):
    idxes = self._get_idxes(idxes)
    obs = [self.envs[i].prev_obs() for i in idxes]
    if convert_batch:
      obs = [convert_batch_with_func(o) for o in zip(*obs)]
    return obs

  def info(self, idxes=None, convert_batch=False):
    idxes = self._get_idxes(idxes)
    info = [self.envs[i].info() for i in idxes]
    if convert_batch:
      info = batch_dicts(info)
    return info

  def prev_output(self, idxes=None, convert_batch=True):
    idxes = self._get_idxes(idxes)
    out = [self.envs[i].prev_output() for i in idxes]
    out = self.process_output(out, convert_batch=convert_batch)
    return out

  def output(self, idxes=None, convert_batch=True):
    idxes = self._get_idxes(idxes)
    out = [self.envs[i].output() for i in idxes]
    out = self.process_output(out, convert_batch=convert_batch)
    return out

  def get_screen(self, size=None, convert_batch=True):
    if hasattr(self.env, 'get_screen'):
      imgs = [env.get_screen() for env in self.envs]
    else:
      imgs = [env.render() for env in self.envs]

    if size is not None:
      # cv2 receive size of form (width, height)
      imgs = [cv2.resize(i, size[::-1], interpolation=cv2.INTER_AREA) 
              for i in imgs]
    if convert_batch:
      imgs = np.stack(imgs)

    return imgs

  def record_default_state(self, aids, states):
    state_type = type(states)
    states = [state_type(*s) for s in zip(*states)]
    for e, a, s in zip(self.envs, aids, states):
      e.record_default_state(a, s)

  def _envvec_op(self, name, convert_batch=True, **kwargs):
    method = lambda e: getattr(e, name)
    if kwargs:
      kwargs = {k: [np.squeeze(x) for x in np.split(v, self.n_envs)]
        if isinstance(v, np.ndarray) else list(zip(*v)) 
        for k, v in kwargs.items()}
      kwargs = [dict(x) for x in zip(*[itertools.product([k], v) 
        for k, v in kwargs.items()])]
      out = [method(env)(**kw) for env, kw in zip(self.envs, kwargs)]
    else:
      out = [method(env)() for env in self.envs]

    return self.process_output(out, convert_batch=convert_batch)

  def close(self):
    if hasattr(self.env, 'close'):
      [env.close() for env in self.envs]


class MASimVecEnv(VecEnvBase):
  def __init__(self, config, env_fn=make_env, agents={}):
    super().__init__(config, env_fn, agents)

  def random_action(self, *args, **kwargs):
    actions = list(zip(*[env.random_action() for env in self.envs]))
    return actions

  def reset(self, idxes=None, convert_batch=True, **kwargs):
    idxes = self._get_idxes(idxes)
    out = [self.envs[i].reset() for i in idxes]

    return self.process_output(out, convert_batch=convert_batch)

  def step(self, actions, convert_batch=True, **kwargs):
    if isinstance(actions, (tuple, list)):
      actions = zip(*actions)
    outs = [e.step(a) for e, a in zip(self.envs, actions)]

    return self.process_output(outs, convert_batch=convert_batch)

  def manual_reset(self):
    [e.manual_reset() for e in self.envs]

  def score(self, idxes=None):
    idxes = self._get_idxes(idxes)
    return [self.envs[i].score() for i in idxes]

  def epslen(self, idxes=None):
    idxes = self._get_idxes(idxes)
    return [self.envs[i].epslen() for i in idxes]

  def mask(self, idxes=None):
    idxes = self._get_idxes(idxes)
    return np.stack([self.envs[i].mask() for i in idxes])

  def game_over(self):
    return np.stack([env.game_over() for env in self.envs])

  def prev_obs(self, idxes=None, convert_batch=True):
    idxes = self._get_idxes(idxes)
    obs = [self.envs[i].prev_obs() for i in idxes]
    if convert_batch:
      obs = batch_dicts(obs)
    return obs

  def info(self, idxes=None, convert_batch=False):
    idxes = self._get_idxes(idxes)
    info = [self.envs[i].info() for i in idxes]
    if convert_batch:
      info = batch_dicts(info)
    return info

  def prev_output(self, idxes=None, convert_batch=True):
    idxes = self._get_idxes(idxes)
    out = [self.envs[i].prev_output() for i in idxes]
    out = self.process_output(out, convert_batch=convert_batch)
    return out

  def output(self, idxes=None, convert_batch=True):
    idxes = self._get_idxes(idxes)
    out = [self.envs[i].output() for i in idxes]

    return self.process_output(out, convert_batch=convert_batch)

  def get_screen(self, size=None, convert_batch=True):
    if hasattr(self.env, 'get_screen'):
      imgs = [env.get_screen() for env in self.envs]
    else:
      imgs = [env.render(mode='rgb_array') for env in self.envs]

    if size is not None:
      # cv2 receive size of form (width, height)
      imgs = [cv2.resize(i, size[::-1], interpolation=cv2.INTER_AREA) 
              for i in imgs]
    if convert_batch:
      imgs = np.stack(imgs)

    return imgs

  def record_default_state(self, aids, states):
    state_type = type(states)
    states = [state_type(*s) for s in zip(*states)]
    for e, a, s in zip(self.envs, aids, states):
      e.record_default_state(a, s)

  def _envvec_op(self, name, convert_batch=True, **kwargs):
    method = lambda e: getattr(e, name)
    if kwargs:
      kwargs = {k: [np.squeeze(x) for x in np.split(v, self.n_envs)]
        if isinstance(v, np.ndarray) else list(zip(*v)) 
        for k, v in kwargs.items()}
      kwargs = [dict(x) for x in zip(*[itertools.product([k], v) 
        for k, v in kwargs.items()])]
      out = [method(env)(**kw) for env, kw in zip(self.envs, kwargs)]
    else:
      out = [method(env)() for env in self.envs]

    return self.process_output(out, convert_batch=convert_batch)

  def close(self):
    if hasattr(self.env, 'close'):
      [env.close() for env in self.envs]


class MATBVecEnv(VecEnvBase):
  """ Different from other Envs which returns data of structure
  (n_envs, n_agents, n_units, ...). MATBVecEnv returns data of
  form (n_agents, n_envs, n_units, ...) since the current players
  vary for different environments
  """
  def __init__(self, config, env_fn=make_env, agents={}):
    super().__init__(config, env_fn, agents)

  def random_action(self, *args, **kwargs):
    actions = [env.random_action() for env in self.envs]
    return self.combine_actions(actions)

  def reset(self, idxes=None, convert_batch=True, **kwargs):
    idxes = self._get_idxes(idxes)
    out = [self.envs[i].reset() for i in idxes]

    return self.process_output(out, convert_batch=convert_batch)

  def step(self, actions, convert_batch=True, **kwargs):
    actions = self.divide_actions(actions)
    outs = [e.step(a) for e, a in zip(self.envs, actions)]

    return self.process_output(outs, convert_batch=convert_batch)

  def manual_reset(self):
    [e.manual_reset() for e in self.envs]

  def score(self, idxes=None):
    idxes = self._get_idxes(idxes)
    return [self.envs[i].score() for i in idxes]

  def epslen(self, idxes=None):
    idxes = self._get_idxes(idxes)
    return [self.envs[i].epslen() for i in idxes]

  def mask(self, idxes=None):
    idxes = self._get_idxes(idxes)
    return np.stack([self.envs[i].mask() for i in idxes])

  def game_over(self):
    return np.stack([env.game_over() for env in self.envs])

  def prev_obs(self, idxes=None, convert_batch=False):
    idxes = self._get_idxes(idxes)
    obs = [self.envs[i].prev_obs() for i in idxes]
    if convert_batch:
      obs = batch_dicts(obs)
    return obs

  def info(self, idxes=None, convert_batch=False):
    idxes = self._get_idxes(idxes)
    info = [self.envs[i].info() for i in idxes]
    if convert_batch:
      info = batch_dicts(info)
    return info

  def prev_output(self, idxes=None, convert_batch=True):
    idxes = self._get_idxes(idxes)
    out = [self.envs[i].prev_output() for i in idxes]
    out = self.process_output(out, convert_batch=convert_batch)
    return out

  def output(self, idxes=None, convert_batch=True):
    idxes = self._get_idxes(idxes)
    out = [self.envs[i].output() for i in idxes]

    return self.process_output(out, convert_batch=convert_batch)

  def get_screen(self, size=None, convert_batch=True):
    if hasattr(self.env, 'get_screen'):
      imgs = [env.get_screen() for env in self.envs]
    else:
      imgs = [env.render(mode='rgb_array') for env in self.envs]

    if size is not None:
      # cv2 receive size of form (width, height)
      imgs = [cv2.resize(i, size[::-1], interpolation=cv2.INTER_AREA) 
              for i in imgs]
    if convert_batch:
      imgs = np.stack(imgs)

    return imgs

  def record_default_state(self, aids, states):
    state_type = type(states)
    states = [state_type(*s) for s in zip(*states)]
    for e, a, s in zip(self.envs, aids, states):
      e.record_default_state(a, s)

  def _envvec_op(self, name, convert_batch=True, **kwargs):
    method = lambda e: getattr(e, name)
    if kwargs:
      kwargs = {k: [np.squeeze(x) for x in np.split(v, self.n_envs)]
        if isinstance(v, np.ndarray) else list(zip(*v)) 
        for k, v in kwargs.items()}
      kwargs = [dict(x) for x in zip(*[itertools.product([k], v) 
        for k, v in kwargs.items()])]
      out = [method(env)(**kw) for env, kw in zip(self.envs, kwargs)]
    else:
      out = [method(env)() for env in self.envs]

    return self.process_output(out, convert_batch=convert_batch)

  def close(self):
    if hasattr(self.env, 'close'):
      [env.close() for env in self.envs]

  def process_output(self, out, convert_batch=True):
    obs = [[] for _ in range(self._stats.n_agents)]
    reward = [[] for _ in range(self._stats.n_agents)]
    discount = [[] for _ in range(self._stats.n_agents)]
    reset = [[] for _ in range(self._stats.n_agents)]
    for eid, o in enumerate(out):
      uid = o.obs['uid']
      o.obs['eid'] = eid
      obs[uid].append(o.obs)
      reward[uid].append(o.reward)
      discount[uid].append(o.discount)
      reset[uid].append(o.reset)
    
    if convert_batch:
      agent_env_outs = [EnvOutput(*o)
        for out in zip(obs, reward, discount, reset)
        for o in [[convert_batch_with_func(o) for o in out]]
      ]
      # Unlike other environments, we return env_output for each agent
      return agent_env_outs
    else:
      return out


if __name__ == '__main__':
  config = dict(
    env_name='mujoco-Ant-v3',
    n_runners=1,
    n_envs=2,
    to_multi_agent=True,
    seed=0
  )
  import numpy as np
  import random
  random.seed(0)
  np.random.seed(0)
  env = VecEnv(config)
  for k in range(10):
    a = env.random_action()
    o, r, d, re = env.step((a))
    print(r)
    if np.any(re):
      eids = [i for i, r in enumerate(re)]
      discounts = d[eids]
      print('reset envs', eids, re, discounts[:, 0], env.epslen(eids))
