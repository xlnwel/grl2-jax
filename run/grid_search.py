import time
import logging
from copy import deepcopy
from multiprocessing import Process

from run.utils import change_config
from utility.utils import product_flatten_dict
logger = logging.getLogger(__name__)


class GridSearch:
    def __init__(self, 
                 configs, 
                 train_func, 
                 n_trials=1, 
                 logdir='logs', 
                 dir_prefix='', 
                 separate_process=False, 
                 delay=1):
        self.configs = configs
        self.train_func = train_func
        self.n_trials = n_trials
        self.root_dir = logdir
        self.dir_prefix = dir_prefix
        self.separate_process = separate_process
        self.delay=delay

        self.processes = []

    def __call__(self, **kwargs):
        self._setup_root_dir()
        if kwargs == {} and self.n_trials == 1 and not self.separate_process:
            # if no argument is passed in, run the default setting
            p = Process(target=self.train_func, args=self.configs)
            self.processes.append(p)
        else:
            # do grid search
            model_name = ''
            self._change_config(model_name, **kwargs)

        return self.processes

    def _setup_root_dir(self):
        if self.dir_prefix:
            self.dir_prefix += '-'
        self.root_dir = (f'{self.root_dir}/'
                        f'{self.configs.env["name"]}/'
                        f'{self.configs.agent["algorithm"]}')

    def _change_config(self, model_name, **kwargs):
        kw_list = product_flatten_dict(**kwargs)
        for d in kw_list:
            # deepcopy to avoid unintended conflicts
            configs = deepcopy(self.configs._asdict())

            for k, v in d.items():
                # search k in configs
                key_configs = {}
                for name, config in configs._asdict():
                    if k in config:
                        key_configs[name] = config
                assert key_configs != [], f'{k} does not appear in any of configs'
                logger.info(f'{k} appears in the following configs: '
                            f'{list([n for n, _ in key_configs.items()])}.\n')
                # change value in config
                for config in key_configs.values():
                    if isinstance(config[k], dict):
                        config[k].update(v)
                    else:
                        config[k] = v
                
                if model_name:
                    model_name += '-'
                # add "key=value" to model name
                model_name += f'{k}={v}'
            
            mn = model_name
            for i in range(1, self.n_trials+1):
                configs = deepcopy(self.configs)

                if self.n_trials > 1:
                    mn += f'-trial{i}' if model_name else f'trial{i}'
                if 'seed' in configs.env:
                    configs.env['seed'] = 1000 * i
                if 'video_path' in configs.env:
                    configs.env['video_path'] = \
                        f'{self.root_dir}/{mn}/{configs.env["video_path"]}'
                
                kw = [f'root_dir={self.root_dir}', f'model_name={mn}']
                change_config(kw, configs)
                p = Process(target=self.train_func, args=configs)
                p.start()
                self.processes.append(p)
                time.sleep(self.delay)   # ensure sub-processs starts in order
