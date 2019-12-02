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
                        action='store_true',
                        help='render the environment. this currently does not work for EnvVec')
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
    elif algorithm == 'ppo2':
        from algo.ppo2.train import main
    elif algorithm == 'sac':
        from algo.sac.train import main
    elif algorithm == 'apex-sac':
        from algo.apex_sac.train import main
    elif algorithm == 'seed-sac':
        from algo.seed_sac.train import main
    elif algorithm == 'dee-sac':
        from algo.dee_sac.train import main
    else:
        raise NotImplementedError

    return main
    
def get_config_file(algorithm):
    if algorithm == 'ppo':
        config_file = 'algo/ppo/config.yaml'
    elif algorithm == 'ppo2':
        config_file = 'algo/ppo2/config.yaml'
    elif algorithm == 'sac':
        config_file = 'algo/sac/config.yaml'
    elif algorithm == 'apex-sac':
        config_file = 'algo/apex_sac/config.yaml'
    elif algorithm == 'seed-sac':
        config_file = 'algo/seed_sac/config.yaml'
    elif algorithm == 'dee-sac':
        config_file = 'algo/dee_sac/config.yaml'
    else:
        raise NotImplementedError

    return config_file

if __name__ == '__main__':
    cmd_args = parse_cmd_args()
    render = cmd_args.render
    processes = []
    if cmd_args.directory != '':
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
        env_config = config['env']
        model_config = config['model']
        agent_config = config['agent']
        buffer_config = config['buffer']
        algo = agent_config['algorithm']
        main = import_main(algo)

        main(env_config, model_config, agent_config, buffer_config, restore=True, render=render)
    else:
        algorithm = list(cmd_args.algorithm)
        for algo in algorithm:
            config_file = get_config_file(algo)
            main = import_main(algo)

            prefix = cmd_args.prefix
            config = load_config(config_file)
            env_config = config['env']
            model_config = config['model']
            agent_config = config['agent']
            buffer_config = config['buffer'] if 'buffer' in config else {}
            if cmd_args.grid_search or cmd_args.trials > 1:
                gs = GridSearch(env_config, model_config, agent_config, buffer_config, 
                                main, render=render, n_trials=cmd_args.trials, dir_prefix=prefix, 
                                separate_process=len(algorithm) > 1)

                # Grid search happens here
                if algo == 'ppo':
                    processes += gs()
                elif algo == 'ppo2':
                    processes += gs(learn_freq=[100, 200])
                elif algo == 'sac':
                    processes += gs()
                elif algo == 'apex-sac':
                    processes += gs()
                elif algo == 'seed-sac':
                    processes += gs()
                else:
                    raise NotImplementedError()
            else:
                if prefix != '':
                    prefix = f'{prefix}-'
                config = load_config(config_file)
                env_config = config['env']
                model_config = config['model']
                agent_config = config['agent']
                buffer_config = config['buffer'] if 'buffer' in config else {}
                agent_config['root_dir'] = f'logs/{prefix}{algo}-{env_config["name"]}'
                env_config['video_path'] = (f'{agent_config["root_dir"]}/'
                                            f'{agent_config["model_name"]}/'
                                            f'{env_config["video_path"]}')
                if len(algorithm) > 1:
                    p = Process(target=main,
                                args=(env_config, 
                                    model_config,
                                    agent_config, 
                                    buffer_config, 
                                    False,
                                    render))
                    p.start()
                    time.sleep(1)
                    processes.append(p)
                else:
                    main(env_config, model_config, agent_config, buffer_config, False, render=render)
    [p.join() for p in processes]
