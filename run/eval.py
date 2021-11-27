from logging import error
import os, sys
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.elements.builder import ElementsBuilder
from core.log import setup_logging
from core.tf_config import *
from utility.display import pwc
from utility.ray_setup import sigint_shutdown_ray
from utility.run import evaluate
from utility.graph import save_video
from utility import pkg
from env.func import create_env
from run.args import parse_eval_args
from run.utils import search_for_config

def build_elements(config, name, env_stats):
    create_model = pkg.import_module(
        name='elements.model', algo=name, place=-1).create_model
    create_actor = pkg.import_module(
        name='elements.actor', algo=name, place=-1).create_actor

    model = create_model(config.model, env_stats, to_build_for_eval=True)
    actor = create_actor(config.actor, model)
    
    return model, actor


def build_strategy(config, name, actor):
    create_strategy = pkg.import_module(
        'elements.strategy', config=config.agent).create_strategy
    strategy = create_strategy(
        name, config.strategy, actor=actor)
    
    return strategy


def build_agent(config, name, strategy):
    create_agent = pkg.import_module(
        'elements.agent', config=config.agent).create_agent

    agent = create_agent(
        config=config.agent, 
        strategy=strategy, 
        name=name,
        to_save_code=False
    )

    return agent


def main(config, n, record=False, size=(128, 128), video_len=1000, 
        fps=30, save=False):
    use_ray = config.env.get('n_workers', 0) > 1
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
    config.env.pop('reward_clip', False)
    if env_name.startswith('procgen') and record:
        config.env['render_mode'] = 'rgb_array'

    env = create_env(config.env, env_fn=make_env)

    env_stats = env.stats()

    builder = ElementsBuilder(config, env_stats, config.algorithm)
    model = builder.build_model(to_build_for_eval=True)
    actor = builder.build_actor(model)
    strategy = builder.build_strategy(actor=actor)
    agent = builder.build_agent(strategy, to_save_code=False)

    if n < env.n_envs:
        n = env.n_envs
    start = time.time()
    scores, epslens, video = evaluate(
        env, agent, n, record_video=record, size=size, video_len=video_len)

    pwc(f'After running {n} episodes',
        f'Score: {np.mean(scores):.3g}',
        f'Epslen: {np.mean(epslens):.3g}', 
        f'Time: {time.time()-start:.3g}',
        color='cyan')

    if record:
        save_video(f'{algo_name}-{env_name}', video, fps=fps)
    if use_ray:
        ray.shutdown()


if __name__ == '__main__':
    args = parse_eval_args()

    setup_logging(args.verbose)

    # load respective config
    config = search_for_config(args.directory)

    silence_tf_logs()
    configure_gpu()
    configure_precision(config.precision)
    
    # get the main function
    try:
        main = pkg.import_main('eval', config=config)
    except:
        print('Default main is used for evaluation')

    # set up env_config
    n = args.n_episodes
    if args.n_workers:
        config.env['n_workers'] = args.n_workers
    if args.n_envs:
        config.env['n_envs'] = args.n_envs
    n = max(args.n_workers * args.n_envs, n)

    main(config, n=n, record=args.record, size=args.size, 
        video_len=args.video_len, fps=args.fps, save=args.save)
