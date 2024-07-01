import os, sys
import argparse
import numpy as np

from tools.pickle import save
from tools.log import do_logging
from core.utils import configure_jax_gpu
from tools.display import print_dict_info
from tools.utils import batch_dicts
from envs.func import create_env
from envs.utils import divide_env_output
from run.ops import *


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('directory',
            type=str,
            default='',
            help='directory where checkpoints and "config.yaml" exist', 
            nargs='*')
  parser.add_argument('--filedir',
            type=str,
            default='datasets')
  parser.add_argument('--n_episodes', '-n',
            type=int,
            default=1)
  parser.add_argument('--n_runners', '-nr',
            type=int,
            default=1)
  parser.add_argument('--n_envs', '-ne',
            type=int,
            default=1)
  args = parser.parse_args()

  return args


def run(agents, env, n_eps):
  data = []
  n = 0
  env_output = env.output()
  while True:
    agent_env_outputs = divide_env_output(env_output)
    actions, stats = zip(*[a(o) for a, o in zip(agents, agent_env_outputs)])
    new_env_output = env.step(actions)
    new_agent_env_outputs = divide_env_output(new_env_output)
    assert len(agent_env_outputs) == len(actions) == len(new_agent_env_outputs)

    next_obs = env.prev_obs()
    dl = [] 
    for i, agent in enumerate(agents):
      d = dict(
        **agent_env_outputs[i].obs, 
        action=actions[i], 
        reward=new_agent_env_outputs[i].reward, 
        discount=new_agent_env_outputs[i].discount, 
        **{f'next_{k}': v for k, v in next_obs[i].items()}, 
        reset=new_agent_env_outputs[i].reset,
      )
      d.update(stats[i])
      dl.append(d)
    data.append(batch_dicts(dl, lambda x: np.concatenate(x, axis=1)))

    env_output = new_env_output
    done_env_ids = [i for i, r in enumerate(env_output.reset[0]) if np.all(r)]
    if done_env_ids:
      n += len(done_env_ids)
      if n >= n_eps:
        break
  return batch_dicts(data, lambda x: np.stack(x, axis=1))


def collect_data(configs, n_eps):
  env = create_env(configs[0].env)
  env_stats = env.stats()
  agents = build_agents(configs, env_stats)
  data = run(agents, env, n_eps)

  return data


def save_data(data, filedir, filename):
  do_logging('Data:')
  print_dict_info(data, '\t')
  save(data, filedir=filedir, filename=filename)


def main(configs, args):
  data = collect_data(configs, args.n_episodes)

  filename = configs[0].env.env_name
  steps = data['obs'].shape[1]
  data_filename = f'{filename}-{steps}'
  do_logging('-'*100)
  save_data(data, filedir=args.filedir, filename=data_filename)
  

if __name__ == '__main__':
  args = parse_args()

  configure_jax_gpu()
  args.n_runners = 1
  args.n_envs = 1
  configs = setup_configs(args)

  main(configs, args)
