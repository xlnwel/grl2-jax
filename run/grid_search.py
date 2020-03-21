import time
from datetime import datetime
from copy import deepcopy
from multiprocessing import Process

from utility.display import assert_colorize


class GridSearch:
    def __init__(self, env_config, model_config, agent_config, buffer_config, 
                train_func, render=False, n_trials=1, dir_prefix='', 
                separate_process=False, delay=1):
        self.env_config = env_config
        self.model_config = model_config
        self.agent_config = agent_config
        self.buffer_config = buffer_config
        self.train_func = train_func
        self.render = render
        self.n_trials = n_trials
        self.dir_prefix = dir_prefix
        self.separate_process = separate_process
        self.delay=delay

        self.processes = []

    def __call__(self, **kwargs):
        self._dir_setup()
        if kwargs == {} and self.n_trials == 1 and not self.separate_process:
            # if no argument is passed in, run the default setting
            self.train_func(self.env_config, self.model_config, self.agent_config, self.buffer_config, False, self.render)        
        else:
            # do grid search
            self.agent_config['model_name'] = 'GS'
            self._change_config(**kwargs)
        return self.processes

    def _dir_setup(self):
        # add date to root directory
        now = datetime.now()
        ans = input('Do you want to add timestamp to directory name?(y/n)\n')
        if ans.lower() == 'y':
            timestamp = f'{now.month:02d}{now.day:02d}-' \
                        f'{now.hour:02d}{now.minute:02d}-'
        else:
            timestamp = ''
        dir_prefix = self.dir_prefix                                 
        self.agent_config['root_dir'] = (f'logs/'
                                        f'{timestamp}'
                                        f'{dir_prefix}'
                                        f'{self.agent_config["algorithm"]}-'
                                        f'{self.env_config["name"]}')

    def _change_config(self, **kwargs):
        if kwargs == {}:
            # basic case
            for i in range(1, self.n_trials+1):
                # arguments should be deep copied here, 
                # otherwise config will be reset if sub-process starts after
                # after the arguments get changed
                env_config = deepcopy(self.env_config)
                model_config = deepcopy(self.model_config)
                agent_config = deepcopy(self.agent_config)
                buffer_config = deepcopy(self.buffer_config)
                if self.n_trials > 1:
                    agent_config['model_name'] += f'/trial{i}'
                env_config['seed'] = 10 * i
                env_config['video_path'] = (f'{agent_config["root_dir"]}/'
                                            f'{agent_config["model_name"]}/'
                                            f'{env_config["video_path"]}')
                p = Process(target=self.train_func,
                            args=(env_config, 
                                model_config,
                                agent_config, 
                                buffer_config, 
                                self.render))
                p.start()
                self.processes.append(p)
                time.sleep(self.delay)   # ensure sub-processs starts in order
        else:
            # recursive case
            kwargs_copy = deepcopy(kwargs)
            key, value = self._popitem(kwargs_copy)

            valid_config = None
            for config in [self.env_config, self.model_config, self.agent_config, self.buffer_config]:
                if key in config:
                    assert_colorize(valid_config is None, f'Conflict: found {key} in both {valid_config} and {config}!')
                    valid_config = config

            err_msg = lambda k, v: f'Invalid Argument: {k}={v}'
            assert_colorize(valid_config is not None, err_msg(key, value))
            if isinstance(value, dict) and len(value) != 0:
                # For simplicity, we do not further consider the case when value is a dict of dicts here
                k, v = self._popitem(value)
                assert_colorize(k in valid_config[key], err_msg(k, v))
                if len(value) != 0:
                    # if there is still something left in value, put value back into kwargs
                    kwargs_copy[key] = value
                self._safe_call(f'-{key}', lambda: self._recursive_trial(valid_config[key], k, v, kwargs_copy))
            else:
                self._recursive_trial(valid_config, key, value, kwargs_copy)

    # helper functions for self._change_config
    def _popitem(self, kwargs):
        assert_colorize(isinstance(kwargs, dict))
        while len(kwargs) != 0:
            k, v = kwargs.popitem()
            if not isinstance(v, list) and not isinstance(v, dict):
                v = [v]
            if len(v) != 0:
                break
        return deepcopy(k), deepcopy(v)

    def _recursive_trial(self, arg, key, value, kwargs):
        assert_colorize(isinstance(value, list), f'Expect value of type list, not {type(value)}: {value}')
        for v in value:
            arg[key] = v
            self._safe_call(f'-{key}={v}', lambda: self._change_config(**kwargs))

    def _safe_call(self, append_name, func):
        old_model_name = self.agent_config['model_name']
        self.agent_config['model_name'] += append_name
        func()
        self.agent_config['model_name'] = old_model_name
