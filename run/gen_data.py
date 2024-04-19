import os, sys
import argparse
import numpy as np
import collections

from core.ckpt.pickle import save
from core.log import do_logging
from core.names import PATH_SPLIT
from core.typing import AttrDict
from core.utils import configure_gpu
from tools.display import print_dict, print_dict_info
from tools.utils import batch_dicts, modify_config
from tools import yaml_op
from env.func import create_env
from env.utils import divide_env_output
from run.utils import *


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
  parser.add_argument('--n_runners', '-nr',
            type=int,
            default=1)
  parser.add_argument('--n_envs', '-ne',
            type=int,
            default=100)
  parser.add_argument('--n_steps', '-ns',
            type=int,
            default=1000)
  parser.add_argument('--from_algo',
            action='store_true', 
            default=False)

  args = parser.parse_args()

  return args


def load_config_from_filedir(filedir, from_algo=False):
  assert filedir.split(PATH_SPLIT)[-1].startswith('a'), filedir
  
  names = filedir.split(PATH_SPLIT)
  algo = names[-5]
  if from_algo:
    algo = names[-5]
    env = names[-6]
    env_suite, env_name = env.split('-')
    filename = env_suite if env_suite == 'ma_mujoco' else env_name
    # load config from algo/*/configs/*.yaml
    config = load_config_with_algo_env(algo, env, filename)
  else:
    # load config from the logging directory
    config = search_for_config(filedir)
  root_dir = os.path.join(names[:-5])
  model_name = os.path.join(names[-5:])
  config = modify_config(
    config, 
    root_dir=root_dir, 
    model_name=model_name, 
    name=algo, 
    seed=int(names[-2][-1])
  )
  # print_dict(config)
  # yaml_path = f'{filedir}/config'
  # yaml_op.save_config(config, path=yaml_path)

  return config


def run(agents, env, n_steps):
  data = []
  env_output = env.output()
  env_stats = env.stats()
  for _ in range(n_steps):
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
      info = env.info(done_env_ids)
      stats = collections.defaultdict(list)
      for i in info:
        for k, v in i.items():
          stats[k].append(v)
      for aid, uids in enumerate(env_stats.aid2uids):
        agent_info = {k: [vv[uids] for vv in v]
            if isinstance(v[0], np.ndarray) else v 
            for k, v in stats.items()}
        agents[aid].store(**agent_info)

  return batch_dicts(data, lambda x: np.stack(x, axis=1))


def collect_data(config, n_steps):
  env = create_env(config.env)
  env_stats = env.stats()
  agents = build_agents(configs, env_stats)
  data = run(agents, env, n_steps=n_steps)
  stats = [agent.get_raw_stats() for agent in agents]
  stats = batch_dicts(stats)

  return data, stats


def save_data(data, filedir, filename):
  do_logging('Data:')
  print_dict_info(data, '\t')
  save(data, filedir=filedir, filename=filename)


def get_stats_path(filename, filedir):
  return f'{filedir}/{filename}-stats.yaml'


def summarize_stats(stats, config):
  for k, v in stats.items():
    stats[k] = np.stack(v)
  # print_dict_info(stats, '\t')

  simple_stats = AttrDict()
  for k in ['score', 'epslen']:
    v = stats[k]
    simple_stats[k] = float(np.mean(v))
    simple_stats[f'{k}_std'] = float(np.std(v))
  simple_stats.root_dir = config.root_dir
  simple_stats.model_name = config.model_name

  filename = config.env.env_name
  env_suite, env_name = filename.split('-')
  simple_stats.algorithm = config.algorithm
  simple_stats.env_suite = env_suite
  simple_stats.env_name = env_name

  return simple_stats


def save_stats(all_stats, stats_path):
  do_logging('Running Statistics:')
  print_dict(all_stats, '\t')
  yaml_op.dump(stats_path, all_stats)


def main(configs, args):
  data, stats = collect_data(configs[0], args.n_steps)
  simple_stats = summarize_stats(stats, configs[0])

  filename = configs[0].env.env_name
  stats_path = get_stats_path(filename, filedir=args.filedir)
  all_stats = yaml_op.load_config(stats_path)
  start = all_stats.get('data_size', 0)
  end = start + data['obs'].shape[0]
  data_filename = f'{filename}-{start}-{end}'
  all_stats[data_filename] = simple_stats
  all_stats.data_size = end

  save_stats(all_stats, stats_path)
  do_logging('-'*100)
  save_data(data, filedir=args.filedir, filename=data_filename)
  

if __name__ == '__main__':
  args = parse_args()

  configure_gpu(None)
  configs = setup_configs(args)
  main(configs, args)
