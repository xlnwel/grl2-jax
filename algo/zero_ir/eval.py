import warnings
warnings.filterwarnings("ignore")
import os, sys
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.elements.builder import ElementsBuilder
from core.tf_config import *
from utility.display import print_dict_info, pwc
from utility.plot import plot_data, plot_data_dict
from utility.ray_setup import sigint_shutdown_ray
from utility.run import evaluate
from utility.utils import dict2AttrDict
from utility.graph import save_video
from utility import pkg
from env.func import create_env


def plot(data: dict, outdir: str, figname: str):
    data = {k: np.squeeze(v) for k, v in data.items()}
    data = {k: np.swapaxes(v, 0, 1) if v.ndim == 2 else v for k, v in data.items()}
    print_dict_info(data)
    plot_data_dict(data, outdir=outdir, figname=figname)
    all_reward = np.concatenate([
        data['reward'], data['meta_reward'], data['trans_reward']])
    plot_data(all_reward, y='reward', outdir=outdir, 
        title=f'{figname}-reward', avg_data=False)

def main(configs, n, record=False, size=(128, 128), video_len=1000, 
        fps=30, out_dir='results', info=''):
    config = dict2AttrDict(configs[0])
    use_ray = config.env.get('n_runners', 0) > 1
    if use_ray:
        import ray
        ray.init()
        sigint_shutdown_ray()

    algo_name = config.algorithm
    env_name = config.env['name']

    try:
        make_env = pkg.import_module('env', algo_name, place=-1).make_env
    except:
        make_env = None
    
    if env_name.startswith('procgen') and record:
        config.env['render_mode'] = 'rgb_array'

    env = create_env(config.env, env_fn=make_env)

    env_stats = env.stats()

    builder = ElementsBuilder(config, env_stats)
    elements = builder.build_acting_agent_from_scratch(to_build_for_eval=True)
    agent = elements.agent
    print('start evaluation')

    if n < env.n_envs:
        n = env.n_envs
    start = time.time()
    scores, epslens, data, video = evaluate(
        env, 
        agent, 
        n, 
        record_video=record, 
        size=size, 
        video_len=video_len
    )

    pwc(f'After running {n} episodes',
        f'Score: {np.mean(scores):.3g}',
        f'Epslen: {np.mean(epslens):.3g}', 
        f'Time: {time.time()-start:.3g}',
        color='cyan')

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
