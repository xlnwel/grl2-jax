import time
import logging
import copy
from multiprocessing import Process
import numpy as np

from run.utils import change_config_with_kw_string
from utility.utils import AttrDict2dict, dict2AttrDict, modify_config, product_flatten_dict

logger = logging.getLogger(__name__)


def find_key(key, config, results):
    if isinstance(config, dict):
        if key in config:
            results.append(config)
        else:
            for v in config.values():
                results = find_key(key, v, results)
    return results


class RandomSearch:
    def __init__(
        self, 
        config, 
        train_func, 
        n_trials=1, 
        logdir='logs', 
        dir_prefix='', 
        separate_process=False, 
        delay=1,
        multiprocess=True
    ):
        self.config = dict2AttrDict(config)
        self.train_func = train_func
        self.n_trials = n_trials
        self.root_dir = logdir
        self.dir_prefix = dir_prefix
        self.separate_process = separate_process
        self.delay=delay

        if multiprocess:
            self.processes = []
        else:
            self.processes = None

    def __call__(self, kw_dict={}, **kwargs):
        self._setup_root_dir()
        kw_dict.update(kwargs)
        for k, v in kw_dict.items():
            if isinstance(v, np.ndarray):
                if np.issubdtype(v.dtype, np.floating):
                    type = float 
                elif np.issubdtype(v.dtype, np.integer): 
                    type = int
                else:
                    raise TypeError(f'Unknown type for {v}: {type}')
                kw_dict[k] = [type(x) for x in v]
            else:
                kw_dict[k] = list(v)
        if kw_dict == {} and self.n_trials == 1 and not self.separate_process:
            # if no argument is passed in, run the default setting
            if self.processes is None:
                self.train_func([self.config])
            else:
                p = Process(target=self.train_func, args=([self.config],))
                self.processes.append(p)
        else:
            # do grid search
            model_name = self.config.model_name
            self._change_config(model_name, kw_dict)

        return self.processes

    def _setup_root_dir(self):
        if self.dir_prefix:
            self.dir_prefix += '-'
        self.root_dir = (
            f'{self.root_dir}/'
            f'{self.config.env["env_name"]}/'
            f'{self.config.algorithm}'
        )

    def _change_config(self, model_name, kwargs):
        seed = kwargs.pop('seed', None)
        if seed is None:
            n_trials = self.n_trials
            seed = list(range(n_trials))
        else:
            n_trials = len(seed)
        assert len(seed) == n_trials, (seed, n_trials)
        kw_list = product_flatten_dict(**kwargs)
        
        for kw in kw_list:
            # deepcopy to avoid unintended conflicts
            config = copy.deepcopy(AttrDict2dict(self.config))
            kw_str = [f'{k}={v}' for k, v in kw.items()]
            new_model_name = change_config_with_kw_string(
                kw_str, config, model_name)

            for i in range(n_trials):
                config2 = dict2AttrDict(config, to_copy=True)
                mn = new_model_name

                if n_trials > 1:
                    mn += f'-seed={i}' if model_name else f'seed={i}'
                if 'video_path' in config2['env']:
                    config2['env']['video_path'] = \
                        f'{self.root_dir}/{mn}/{config2["env"]["video_path"]}'
                
                config2 = modify_config(
                    config2, 
                    root_dir=self.root_dir, 
                    model_name=mn, 
                    seed=seed[i]
                )
                # print_dict(config2)
                if self.processes is None:
                    self.train_func([config2])
                else:
                    p = Process(target=self.train_func, args=([config2],))
                    p.start()
                    self.processes.append(p)
                    time.sleep(self.delay)   # ensure sub-processs starts in order
