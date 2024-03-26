import os
import re
import random
import jax
import jax.numpy as jnp

from core.names import MODEL, PATH_SPLIT
from core.typing import ModelPath, retrieve_model_path
from core.log import do_logging
from nn.utils import reset_linear_weights, reset_weights


def divide_runners(n_agents, n_runners, online_frac):
  if n_runners < n_agents:
    return n_runners, 0
  n_agent_runners = int(n_runners * (1 - online_frac) // n_agents)
  n_online_runners = n_runners - n_agent_runners * n_agents
  assert n_agent_runners * n_agents + n_online_runners == n_runners, \
    (n_agent_runners, n_agents, n_online_runners, n_runners)
  return n_online_runners, n_agent_runners


def reset_policy_head(weights, config):
  def reset_logstd(vv, rng):
    if config.model.policy.sigmoid_scale:
      logstd = config.model.policy.std_x_coef
      logstd = jnp.minimum(logstd, vv*2)
    else:
      logstd = jax.lax.log(config.model.policy.init_std)
      logstd = jnp.minimum(logstd, vv/2)
    return reset_weights(vv, rng, 'constant', value=logstd)
  
  rng = jax.random.PRNGKey(random.randint(0, 2**32))
  if 'policies' in weights[MODEL]:
    for policy in weights[MODEL]['policies']:
      for k, v in policy.items():
        if k.startswith('policy/head'):
          policy[k] = reset_linear_weights(
            v, rng, 'orthogonal', scale=.01)
          do_logging(f'{k} is reset', color='green')
        elif k == 'policy':
          for kk, vv in policy[k].items():
            if kk.endswith('logstd'):
              policy[k][kk] = reset_logstd(vv, rng)
              do_logging(f'{k}.{kk} is reset', color='green')
  elif 'policy' in weights[MODEL]:
    policy = weights[MODEL]['policy']
    for k, v in policy.items():
      if k.startswith('policy/head'):
        policy[k] = reset_linear_weights(
          v, rng, 'orthogonal', scale=.01)
        do_logging(f'{k} is reset', color='green')
      elif k == 'policy':
        for kk, vv in policy[k].items():
          if kk.endswith('logstd'):
            policy[k][kk] = reset_logstd(vv, rng)
            do_logging(f'{k}.{kk} is reset', color='green')
  return weights


def find_latest_model(path):
  _, a = path.rsplit(PATH_SPLIT, 1)
  assert a.startswith('a') and a[1:].isdigit(), path
  max_iid = 0
  latest_model = ''
  if not os.path.isdir(path):
    return None
  for d in os.listdir(path):
    if d.startswith('i'):
      iid = int(d.split('-')[0][1:])
      if iid > max_iid:
        max_iid = iid
        latest_model = os.path.join(path, d)
  if latest_model == '':
    raise ValueError(path)
  return retrieve_model_path(latest_model)


def find_all_models(path, pattern='.*'):
  models = []
  for root, dirs, files in os.walk(path):
    if 'src' in root:
      continue
    if not re.match(pattern, root):
      continue
    for d in dirs:
      if d.startswith('a') and d[1:].isdigit():
        agent_dir = os.path.join(root, d)
        for dd in os.listdir(agent_dir):
          if dd.startswith('i'):
            path = os.path.join(agent_dir, dd)
            models.append(retrieve_model_path(path))
  return models
