import os, sys
import time
import itertools
from multiprocessing import Process
from copy import deepcopy
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run.grid_search import GridSearch
from utility.utils import str2bool, eval_str
from utility.yaml_op import load_config
from utility.display import assert_colorize, pwc
from run.cmd_args import parse_cmd_args


def import_main(algorithm):
    if algorithm == 'ppo':
        from algo.ppo.train import main
    elif algorithm == 'ppo2':
        from algo.ppo2.train import main
    elif algorithm == 'sac':
        from algo.sac.train import main
    elif algorithm == 'sacar':
        from algo.sacar.train import main
    elif algorithm == 'd3qn':
        from algo.d3qn.train import main
    elif algorithm.startswith('apex-dr'):
        from algo.apex_dr.train import main
    elif algorithm.startswith('apex-ar'):
        from algo.apex_ar.train import main
    elif algorithm.startswith('apex') or algorithm.startswith('asap'):
        from algo.apex.train import main
    elif algorithm == 'seed-sac':
        from algo.seed_sac.train import main
    elif algorithm.startswith('dew'):
        from algo.dew.train import main
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
    elif algorithm == 'sacar':
        config_file = 'algo/sacar/config.yaml'
    elif algorithm == 'sac-il':
        config_file = 'algo/sac_il/config.yaml'
    elif algorithm == 'd3qn':
        config_file = 'algo/d3qn/config.yaml'
    elif algorithm == 'apex-d3qn':
        config_file = 'algo/apex/d3qn_config.yaml'
    elif algorithm == 'apex-sac' or algorithm == 'apex-t-sac':
        config_file = 'algo/apex/sac_config.yaml'
    elif algorithm == 'apex-dr-sac':
        config_file = 'algo/apex_dr/sac_config.yaml'
    elif algorithm == 'apex-ar-sac':
        config_file = 'algo/apex_ar/sac_config.yaml'
    elif algorithm.startswith('asap'):
        if algorithm.endswith('sac'):
            config_file = 'algo/asap/sac_config.yaml'
    elif algorithm == 'asap2-sac' or algorithm == 'asap2-t-sac':
        config_file = 'algo/asap2/sac_config.yaml'
    elif algorithm == 'asap-d3qn':
        config_file = 'algo/asap/d3qn_config.yaml'
    elif algorithm == 'seed-sac':
        config_file = 'algo/seed/config.yaml'
    elif algorithm == 'dew-sac':
        config_file = 'algo/dew/sac_config.yaml'
    elif algorithm == 'dew-d3qn':
        config_file = 'algo/dew/d3qn_config.yaml'
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
        replay_config = config.get('buffer') or config.get('replay')
        algo = agent_config['algorithm']
        main = import_main(algo)

        main(env_config, model_config, agent_config, replay_config, restore=True, render=render)
    else:
        algorithm = list(cmd_args.algorithm)
        environment = list(cmd_args.environment)
        algo_env = list(itertools.product(algorithm, environment))
        for algo, env in algo_env:
            config_file = get_config_file(algo)
            main = import_main(algo)

            prefix = cmd_args.prefix
            config = load_config(config_file)
            env_config = config['env']
            model_config = config['model']
            agent_config = config['agent']
            replay_config = config.get('buffer') or config.get('replay')
            agent_config['algorithm'] = algo
            if env:
                env_config['name'] = env
            if cmd_args.max_steps:
                agent_config['MAX_STEPS'] = cmd_args.max_steps
            if cmd_args.n_envs:
                env_config['n_envs'] = cmd_args.n_envs
            if cmd_args.grid_search or cmd_args.trials > 1:
                gs = GridSearch(env_config, model_config, agent_config, replay_config, 
                                main, render=render, n_trials=cmd_args.trials, dir_prefix=prefix, 
                                separate_process=len(algo_env) > 1, delay=cmd_args.delay)

                if cmd_args.grid_search:
                    # Grid search happens here
                    if algo == 'ppo':
                        processes += gs()
                    elif algo == 'ppo2':
                        processes += gs(value_coef=[0.01, 0.001])
                    elif algo == 'sac':
                        processes += gs(temperature=dict(value=[0.01, 0.2]))
                    elif algo == 'sacar':
                        processes += gs(actor=dict(max_ar=[5, 10]))
                    elif algo == 'd3qn':
                        processes += gs()
                    else:
                        raise NotImplementedError()
                else:
                    processes += gs()
            else:
                if prefix != '':
                    prefix = f'{prefix}-'

                if cmd_args.kwargs:
                    for s in cmd_args.kwargs:
                        key, value = s.split('=')
                        value = eval_str(value)
                        if prefix == '':
                            prefix = s
                        else:
                            prefix += '-' + s

                        # change kwargs in config
                        valid_config = None
                        for config in [env_config, model_config, agent_config, replay_config]:
                            if key in config:
                                assert_colorize(valid_config is None, f'Conflict: found {key} in both {valid_config} and {config}!')
                                valid_config = config
                        valid_config[key]  = value


                agent_config['root_dir'] = f'logs/{prefix}{algo}-{env_config["name"]}'
                env_config['video_path'] = (f'{agent_config["root_dir"]}/'
                                            f'{agent_config["model_name"]}/'
                                            f'{env_config["video_path"]}')
                if len(algorithm) > 1:
                    p = Process(target=main,
                                args=(env_config, 
                                    model_config,
                                    agent_config, 
                                    replay_config, 
                                    False,
                                    render))
                    p.start()
                    time.sleep(1)
                    processes.append(p)
                else:
                    main(env_config, model_config, agent_config, replay_config, False, render=render)
    [p.join() for p in processes]
