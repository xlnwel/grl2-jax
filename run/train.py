import os, sys
import time
import argparse
import logging
from multiprocessing import Process
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run.grid_search import GridSearch
from utility.utils import str2bool
from utility.yaml_op import load_config
from utility.display import assert_colorize, pwc



def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', '-a',
                        type=str,
                        nargs='*')
    parser.add_argument('--environment', '-e',
                        type=str,
                        default='')
    parser.add_argument('--render', '-r',
                        action='store_true')
    parser.add_argument('--trials', '-t',
                        type=int,
                        default=1,
                        help='number of trials')
    parser.add_argument('--prefix', '-p',
                        default='',
                        help='prefix for model dir')
    parser.add_argument('--directory', '-d',
                        type=str,
                        default='',
                        help='directory where checkpoints and "config.yaml" exist')
    parser.add_argument('--grid_search', '-gs',
                        action='store_true')
    args = parser.parse_args()

    return args

def import_main(algorithm):
    if algorithm == 'ppo':
        from algo.ppo.train import main
    elif algorithm == 'sac':
        from algo.sac.train import main
    else:
        raise NotImplementedError

    return main
    
def get_arg_file(algorithm):
    if algorithm == 'ppo':
        arg_file = 'algo/ppo/config.yaml'
    elif algorithm == 'sac':
        arg_file = 'algo/sac/config.yaml'
    else:
        raise NotImplementedError

    return arg_file

if __name__ == '__main__':
    cmd_args = parse_cmd_args()
    algorithm = list(cmd_args.algorithm)

    processes = []
    for algo in algorithm:
        arg_file = get_arg_file(algo)
        main = import_main(algo)
        
        render = cmd_args.render

        if cmd_args.directory != '':
            # load configuration
            config = load_config(arg_file)
            env_config = config['env']
            model_config = config['model']
            agent_config = config['agent']
            buffer_config = config['buffer'] if 'buffer' in config else {}
            # load model and log path
            directory = cmd_args.directory
            config_file = None
            for root, _, files in os.walk(directory):
                for f in files:
                    if f == 'config.yaml' and config_file is None:
                        config_file = os.path.join(root, f)
                        break
                    elif f =='config.yaml' and config_file is not None:
                        pwc(f'Get multiple "config.yaml": "{config_file}" and "{os.path.join(root, f)}"')
                        sys.exit()

            config = load_config(config_file)
            agent_config['model_root_dir'] = config['model_root_dir']
            agent_config['log_root_dir'] = config['log_root_dir']
            agent_config['model_name'] = config['model_name']

            main(env_config, model_config, agent_config, buffer_config, render=render)
        else:
            prefix = cmd_args.prefix
            if cmd_args.grid_search:
                gs = GridSearch(arg_file, main, render=render, 
                                n_trials=cmd_args.trials, dir_prefix=prefix, 
                                separate_process=len(algorithm) > 1)

                # Grid search happens here
                if algo == 'ppo':
                    processes += gs()
                elif algo == 'sac':
                    processes += gs()
                else:
                    raise NotImplementedError()
            else:
                if prefix == '':
                    prefix = 'baseline'
                config = load_config(arg_file)
                env_config = config['env']
                model_config = config['model']
                agent_config = config['agent']
                buffer_config = config['buffer'] if 'buffer' in config else {}
                dir_fn = lambda filename: f'logs/{prefix}-{algo}-{env_config["name"]}/{filename}'
                for root_dir in ['model_root_dir', 'log_root_dir']:
                    agent_config[root_dir] = dir_fn(agent_config[root_dir])
                env_config['video_path'] = dir_fn(env_config['video_path'])
                if len(algorithm) > 1:
                    p = Process(target=main,
                                args=(env_config, 
                                    model_config,
                                    agent_config, 
                                    buffer_config, 
                                    render))
                    p.start()
                    time.sleep(1)
                    processes.append(p)
                else:
                    main(env_config, model_config, agent_config, buffer_config, render)
    [p.join() for p in processes]
