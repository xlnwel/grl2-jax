import warnings
warnings.filterwarnings("ignore")

import os, sys
os.environ['XLA_FLAGS'] = "--xla_gpu_force_compilation_parallelism=1"

import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.elements.builder import ElementsBuilder
from core.log import setup_logging, do_logging
from core.names import PATH_SPLIT
from core.typing import dict2AttrDict
from core.utils import configure_gpu
from tools.plot import plot_data_dict
from tools.ray_setup import sigint_shutdown_ray
from tools.run import evaluate
from tools.graph import save_video
from tools import pkg
from env.func import create_env
from run.args import parse_eval_args
from run.utils import setup_configs, compute_episodes


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
  for i, score in enumerate(scores):
    do_logging(f'\tAgent{i} Score: {np.mean(score):.3g}', color='cyan')
  do_logging(f'\tEpslen: {np.mean(epslens):.3g}', color='cyan')
  do_logging(f'\tTime: {time.time()-start:.3g}', color='cyan')

  filename = os.path.join(out_dir, f'{algo_name}-{env_name}', config["model_name"])
  out_dir, filename = filename.rsplit(PATH_SPLIT, maxsplit=1)
  if info != "" and info is not None:
    filename = os.path.join(out_dir, filename, info)
    out_dir, filename = filename.rsplit(PATH_SPLIT, maxsplit=1)
  if record:
    plot(data, out_dir, filename)
    save_video(filename, video, fps=fps, out_dir=out_dir)
  if use_ray:
    ray.shutdown()

  return scores, epslens, video


if __name__ == '__main__':
  args = parse_eval_args()

  configure_gpu()
  setup_logging(args.verbose)
  configs = setup_configs(args)
  n = compute_episodes(args)


  main(configs, n=n, record=args.record, size=args.size, 
    video_len=args.video_len, fps=args.fps, 
    info=args.info)
