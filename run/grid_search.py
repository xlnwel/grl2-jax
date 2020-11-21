import time
from datetime import datetime
from copy import deepcopy
from multiprocessing import Process
import numpy as np


class GridSearch:
    def __init__(self, env_config, model_config, agent_config, replay_config, 
                train_func, n_trials=1, logdir='logs', separate_process=False, delay=1):
        self.env_config = env_config
        self.model_config = model_config
        self.agent_config = agent_config
        self.replay_config = replay_config
        self.train_func = train_func
        self.n_trials = n_trials
        self.logdir = logdir
        self.separate_process = separate_process
        self.delay=delay

        self.processes = []

    def __call__(self, **kwargs):
        self._dir_setup()
        if kwargs == {} and self.n_trials == 1 and not self.separate_process:
            # if no argument is passed in, run the default setting
            self.train_func(self.env_config, self.model_config, self.agent_config, self.replay_config)        
        else:
            # do grid search
            self.agent_config['model_name'] = ''
            self._change_config(**kwargs)

        return self.processes

    def _dir_setup(self):
        # add date to root directory
        now = datetime.now()
        timestamp = ''
        # ans = input('Do you want to add timestamp to directory name?(y/n)\n')
        # if ans.lower() == 'y':
        #     timestamp = f'{now.month:02d}{now.day:02d}-' \
        #                 f'{now.hour:02d}{now.minute:02d}-'                                
        self.agent_config['root_dir'] = (f'{self.logdir}/'
                                        f'{self.env_config["name"]}/'
                                        f'{self.agent_config["algorithm"]}')

    def _change_config(self, **kwargs):
        if kwargs == {}:
            # basic case
            for i in range(1, self.n_trials+1):
                # arguments should be deep copied here, 
                # otherwise config will be reset if sub-process starts
                # after the arguments get changed
                env_config = deepcopy(self.env_config)
                model_config = deepcopy(self.model_config)
                agent_config = deepcopy(self.agent_config)
                replay_config = deepcopy(self.replay_config)
                if self.n_trials > 1:
                    agent_config['model_name'] += f'-trial{i}' if agent_config['model_name'] else f'trial{i}'
                if 'seed' in env_config:
                    env_config['seed'] = 1000 * i
                if 'video_path' in env_config:
                    env_config['video_path'] = (f'{agent_config["root_dir"]}/'
                                                f'{agent_config["model_name"]}/'
                                                f'{env_config["video_path"]}')
                p = Process(target=self.train_func,
                            args=(env_config, 
                                model_config,
                                agent_config, 
                                replay_config))
                p.start()
                self.processes.append(p)
                time.sleep(self.delay)   # ensure sub-processs starts in order
        else:
            # recursive case
            kwargs_copy = deepcopy(kwargs)
            key, value = self._popitem(kwargs_copy)

            configs = []
            for name, config in zip(['env', 'model', 'agent', 'replay'],
                            [self.env_config, self.model_config, self.agent_config, self.replay_config]):
                if key in config:
                    configs.append((name, config))

            if len(configs) > 1:
                print(f'Warning: {key} appears in the following configs: '
                        f'{list([n for n, _ in configs])}.\n')
                configs = [c for _, c in configs]
            else:
                configs = [c for _, c in configs]

            err_msg = lambda k, v: f'Invalid Argument: {k}={v}'
            assert configs != [], err_msg(key, value)
            if isinstance(value, dict) and len(value) != 0:
                # For simplicity, we do not further consider the case when value is a dict of dicts here
                k, v = self._popitem(value)
                for c in configs:
                    assert k in c[key], err_msg(k, v)
                if len(value) != 0:
                    # if there is still something left in value, put value back into kwargs
                    kwargs_copy[key] = value
                sub_configs = [c[key] for c in configs]
                self._safe_call(f'{key}', lambda: self._recursive_trial(sub_configs, k, v, kwargs_copy))
            else:
                self._recursive_trial(configs, key, value, kwargs_copy)

    # helper functions for self._change_config
    def _popitem(self, kwargs):
        assert isinstance(kwargs, dict)
        while len(kwargs) != 0:
            k, v = kwargs.popitem()
            if isinstance(v, np.ndarray):
                v = list(v)
            elif not isinstance(v, list) \
                and not isinstance(v, dict):
                v = [v]
            if len(v) != 0:
                break
        return deepcopy(k), deepcopy(v)

    def _recursive_trial(self, configs, key, value, kwargs):
        assert isinstance(value, list), \
            f'Expect value of type list or np.ndarray, not {type(value)}: {value}'
        for v in value:
            for c in configs:
                c[key] = v
            self._safe_call(f'{key}={v}', lambda: self._change_config(**kwargs))
            
    def _safe_call(self, append_name, func):
        """ safely append 'append_name' to 'model_name' in 'agent_config' and call func """
        old_model_name = self.agent_config['model_name']
        self.agent_config['model_name'] += f'-{append_name}' if old_model_name else append_name
        func()
        self.agent_config['model_name'] = old_model_name
