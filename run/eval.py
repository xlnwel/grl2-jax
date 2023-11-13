import warnings
warnings.filterwarnings("ignore")

import os, sys
os.environ['XLA_FLAGS'] = "--xla_gpu_force_compilation_parallelism=1"

import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.elements.builder import ElementsBuilder
from core.log import setup_logging, do_logging
from core.typing import dict2AttrDict
from core.utils import configure_gpu
from tools.plot import plot_data_dict
from tools.ray_setup import sigint_shutdown_ray
from tools.run import evaluate
from tools.graph import save_video
from tools.utils import modify_config
from tools import pkg
from env.func import create_env
from run.args import parse_eval_args
from run.utils import search_for_config, search_for_all_configs


def plot(data: dict, outdir: str, figname: str):
  data = {k: np.squeeze(v) for k, v in data.items()}
  data = {k: np.swapaxes(v, 0, 1) if v.ndim == 2 else v for k, v in data.items()}
  plot_data_dict(data, outdir=outdir, figname=figname)

def main(configs, n, record=False, size=(128, 128), video_len=1000, 
    fps=30, out_dir='results', info=''):
  config = dict2AttrDict(configs[0])
  use_ray = config.env.get('n_runners', 0) > 1
  if use_ray:
    import ray
    ray.init()
    sigint_shutdown_ray()

  algo_name = config.algorithm
  env_name = config.env['env_name']

  try:
    make_env = pkg.import_module('env', algo=algo_name, place=-1).make_env
  except Exception as e:
    make_env = None
  
  if env_name.startswith('procgen') and record:
    config.env['render_mode'] = 'rgb_array'

  env = create_env(config.env, env_fn=make_env)

  env_stats = env.stats()

  agents = []
  for config in configs:
    builder = ElementsBuilder(config, env_stats)
    elements = builder.build_acting_agent_from_scratch(to_build_for_eval=True)
    agents.append(elements.agent)
  print('start evaluation')

  if n < env.n_envs:
    n = env.n_envs
  start = time.time()
  scores, epslens, data, video = evaluate(
    env, 
    agents, 
    n, 
    record_video=record, 
    size=size, 
    video_len=video_len
  )

  do_logging(f'After running {n} episodes', color='cyan')
  for i, (score, epslen) in enumerate(zip(scores, epslens)):
    do_logging(f'\tScore for Agent{i}: {np.mean(score):.3g}\n', color='cyan')
    do_logging(f'\tEpslen for Agent{i}: {np.mean(epslen):.3g}\n', color='cyan')
  do_logging(f'\tTime: {time.time()-start:.3g}', color='cyan')

  filename = f'{out_dir}/{algo_name}-{env_name}/{config["model_name"]}'
  out_dir, filename = filename.rsplit('/', maxsplit=1)
  if info != "" and info is not None:
    filename = f'{out_dir}/{filename}/{info}'
    out_dir, filename = filename.rsplit('/', maxsplit=1)
  if record:
    plot(data, out_dir, filename)
    save_video(filename, video, fps=fps, out_dir=out_dir)
  if use_ray:
    ray.shutdown()

  return scores, epslens, video


if __name__ == '__main__':
  args = parse_eval_args()

  setup_logging(args.verbose)

  # load respective config
  if len(args.directory) == 1:
    configs = search_for_all_configs(args.directory[0])
    directories = [args.directory[0] for _ in configs]
  else:
    configs = [search_for_config(d) for d in args.directory]
    directories = args.directory
  config = configs[0]

  # get the main function
  # try:
  #   main = pkg.import_main('eval', config=config)
  # except Exception as e:
  #   do_logging(f'Default evaluation is used due to error: {e}', color='red')

  configure_gpu()

  # set up env_config
  for d, config in zip(directories, configs):
    if not d.startswith(config.root_dir):
      i = d.find(config.root_dir)
      if i == -1:
        names = d.split('/')
        root_dir = '/'.join([n for n in names if n not in config.model_name])
        model_name = '/'.join([n for n in names if n in config.model_name])
        model_name = config.model_name[config.model_name.find(model_name):]
      else:
        root_dir = d[:i] + config.root_dir
        model_name = config.model_name
      do_logging(f'root dir: {root_dir}')
      do_logging(f'model name: {model_name}')
      config = modify_config(
        config, 
        overwrite_existed_only=True, 
        root_dir=root_dir, 
        model_name=model_name
      )
    n = args.n_episodes
    if args.n_runners:
      if 'runner' in config:
        config.runner.n_runners = args.n_runners
      config.env.n_runners = args.n_runners
    if args.n_envs:
      config.env.n_envs = args.n_envs
    n = max(args.n_runners * args.n_envs, n)

  main(configs, n=n, record=args.record, size=args.size, 
    video_len=args.video_len, fps=args.fps, 
    info=args.info)
