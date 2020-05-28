import os, sys
import importlib
import argparse
import logging
from copy import deepcopy
import numpy as np
import ray

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.tf_config import *
from utility.display import pwc
from utility.yaml_op import load_config
from utility.ray_setup import sigint_shutdown_ray
from utility.run import evaluate
from utility.graph import save_video
from utility import pkg
from env.gym_env import create_env


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',
                        type=str,
                        help='directory where checkpoints and "config.yaml" exist')
    parser.add_argument('--record', '-r',
                        action='store_true')
    parser.add_argument('--n_episodes', '-n', type=int, default=1)
    parser.add_argument('--n_envs', '-ne', type=int, default=0)
    parser.add_argument('--n_workers', '-nw', type=int, default=0)
    parser.add_argument('--size', '-s', nargs='+', type=int, default=[128, 128])
    args = parser.parse_args()

    return args

def main(env_config, model_config, agent_config, n, record=False, size=(128, 128)):
    silence_tf_logs()
    configure_gpu()
    configure_precision(agent_config['precision'])

    use_ray = env_config.get('n_workers', 0) > 1
    if use_ray:
        ray.init()
        sigint_shutdown_ray()

    algo_name = agent_config['algorithm']
    env_name = env_config['name']

    env = create_env(env_config)
    create_model, Agent = pkg.import_agent(config=agent_config)    
    models = create_model(model_config, env)

    agent = Agent( 
        config=agent_config, 
        models=models, 
        dataset=None, 
        env=env)

    scores, epslens, video = evaluate(env, agent, n, record=record, size=size)
    if record:
        save_video(f'{algo_name}-{env_name}', video)
    pwc(f'After running {n} episodes',
        f'Score: {np.mean(scores)}\tEpslen: {np.mean(epslens)}', color='cyan')

    if use_ray:
        ray.shutdown()

if __name__ == '__main__':
    args = parse_cmd_args()

    # search for config.yaml
    directory = args.directory
    config_file = None
    for root, _, files in os.walk(directory):
        for f in files:
            if 'src' in root:
                break
            elif f.endswith('config.yaml') and config_file is None:
                config_file = os.path.join(root, f)
                break
            elif f.endswith('config.yaml') and config_file is not None:
                pwc(f'Get multiple "config.yaml": "{config_file}" and "{os.path.join(root, f)}"')
                sys.exit()

    # load respective config
    config = load_config(config_file)
    env_config = config['env']
    model_config = config['model']
    agent_config = config['agent']

    # get the main function
    try:
        main = pkg.import_main('eval', config=agent_config)
    except:
        print('Default main is used for evaluation')
    record = args.record

    # set up env_config
    n = args.n_episodes
    if args.n_workers:
        env_config['n_workers'] = args.n_workers
    if args.n_envs:
        env_config['n_envs'] = args.n_envs
    n = max(args.n_workers * args.n_envs, n)
    env_config['seed'] = np.random.randint(1000)
    
    main(env_config, model_config, agent_config, n=n, record=record, size=tuple(args.size))
