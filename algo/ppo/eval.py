import warnings
warnings.filterwarnings("ignore")
import os, sys
import time
import collections
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.elements.builder import ElementsBuilder
from tools.log import do_logging
from core.names import PATH_SPLIT
from core.utils import configure_jax_gpu
from core.typing import dict2AttrDict, get_basic_model_name
from tools.display import print_dict_info
from tools.plot import plot_data, plot_data_dict
from tools.ray_setup import sigint_shutdown_ray
from tools.graph import save_video
from tools.utils import batch_dicts, flatten_dict
from tools import pkg
from env.func import create_env
from env.typing import EnvOutput


def evaluate(
  env, 
  agents, 
  n, 
  record_video=True, 
  size=(128, 128), 
  video_len=1000, 
  n_windows=4
):
  for a in agents:
    a.strategy.model.check_params(False)

  n_done_eps = 0
  n_run_eps = env.n_envs
  scores = []
  epslens = []
  frames = [collections.deque(maxlen=video_len) 
    for _ in range(min(n_windows, env.n_envs))]
  stats_list = []

  prev_done = np.zeros(env.n_envs)
  env.manual_reset()
  env_output = env.reset()
  env_outputs = [EnvOutput(*o) for o in zip(*env_output)]
  while n_done_eps < n:
    if record_video:
      lka = env.get_screen(size=size)
      if env.env_type == 'Env':
        frames[0].append(lka)
      else:
        for i in range(len(frames)):
          frames[i].append(lka[i])

    acts, stats = zip(*[a(eo, evaluation=True) for a, eo in zip(agents, env_outputs)])

    new_stats = {}
    for i, s in enumerate(stats):
      for k, v in s.items():
        new_stats[f'{k}_{i}'] = v

    action = np.concatenate(acts, axis=1)
    env_output = env.step(action)
    env_outputs = [EnvOutput(*o) for o in zip(*env_output)]
    for i, eo in enumerate(env_outputs):
      new_stats[f'reward_{i}'] = eo.reward
    stats_list.append(new_stats)

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


def plot(data: dict, outdir: str, figname: str):
  data = flatten_dict(data)
  data = {k: np.squeeze(v) for k, v in data.items() if v is not None}
  # data = {k: np.swapaxes(v, 0, 1) for k, v in data.items() if v.ndim == 2}
  plot_data_dict(data, outdir=outdir, figname=figname)
  # reward = data['reward']
  # plot_data(reward, y='reward', outdir=outdir, 
  #   title=f'{figname}-reward', avg_data=False)


def main(configs, n, record=False, size=(256, 256), video_len=1000, 
    fps=30, out_dir=None, info=''):
  configure_jax_gpu()

  configs = [dict2AttrDict(config, to_copy=True) for config in configs]
  config = configs[0]

  # build environment
  use_ray = config.env.get('n_runners', 0) > 1
  if use_ray:
    import ray
    ray.init()
    sigint_shutdown_ray()

  algo_name = config.algorithm
  env_name = config.env['name']
  do_logging(f"algo name: {algo_name}")

  try:
    make_env = pkg.import_module('env', algo_name, place=-1).make_env
  except:
    make_env = None
  
  env = create_env(config.env, env_fn=make_env)
  env_stats = env.stats()

  # build acting agents
  agents = []
  for config in configs:
    config = dict2AttrDict(config, to_copy=True)
    builder = ElementsBuilder(config, env_stats, to_save_code=False)
    elements = builder.build_acting_agent_from_scratch(to_build_for_eval=True)
    agents.append(elements.agent)
  do_logging('Start evaluation')

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
  do_logging(f'\tScore: {np.mean(scores):.3g}\n', color='cyan')
  do_logging(f'\tEpslen: {np.mean(epslens):.3g}\n', color='cyan')
  do_logging(f'\tTime: {time.time()-start:.3g}', color='cyan')

  if out_dir is None:
    model_name = get_basic_model_name(config.model_name)
    do_logging(f'model name: {model_name}')
    filename = f'{config.root_dir}/{model_name}'
    do_logging(f'filename: {filename}')
  else:
    filename = f'{out_dir}/{algo_name}-{env_name}/{config["model_name"]}'
  filename = filename.replace('/', PATH_SPLIT)
  out_dir, filename = filename.rsplit(PATH_SPLIT, maxsplit=1)
  if info != "" and info is not None:
    filename = f'{out_dir}/{filename}/{info}'
    filename = filename.replace('/', PATH_SPLIT)
    out_dir, filename = filename.rsplit(PATH_SPLIT, maxsplit=1)
  if record:
    plot(data, out_dir, filename)
    save_video(filename, video, fps=fps, out_dir=out_dir)
  if use_ray:
    ray.shutdown()
  
  do_logging('Evaluation completed')
  return scores, epslens, video
