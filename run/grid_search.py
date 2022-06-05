import time
import logging
import copy
from multiprocessing import Process

from run.utils import change_config
from utility.display import print_dict
from utility.utils import AttrDict2dict, dict2AttrDict, modify_config, product_flatten_dict
logger = logging.getLogger(__name__)


def find_key(key, config, results):
    if isinstance(config, dict):
        if key in config:
            results.append(config)
        else:
            for v in config.values():
                find_key(key, v, results)
    return results


class GridSearch:
    def __init__(
        self, 
        config, 
        train_func, 
        n_trials=1, 
        logdir='logs', 
        dir_prefix='', 
        separate_process=False, 
        delay=1
    ):
        self.config = dict2AttrDict(config)
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
            p = Process(target=self.train_func, args=([self.config],))
            self.processes.append(p)
        else:
            # do grid search
            model_name = self.config.model_name
            self._change_config(model_name, **kwargs)

        return self.processes

    def _setup_root_dir(self):
        if self.dir_prefix:
            self.dir_prefix += '-'
        self.root_dir = (
            f'{self.root_dir}/'
            f'{self.config.env["env_name"]}/'
            f'{self.config.algorithm}'
        )

    def _change_config(self, model_name, **kwargs):
        kw_list = product_flatten_dict(**kwargs)
        
        for kw in kw_list:
            # deepcopy to avoid unintended conflicts
            config = copy.deepcopy(AttrDict2dict(self.config))
            mn = copy.copy(model_name)
            for k, v in kw.items():
                # search k in config
                results = find_key(k, config, [])
                assert results != [], f'{k} does not appear in any of config'
                logger.info(f'{k} appears in the following config: '
                            f'{list(results)}.\n')
                # change value in config
                for d in results:
                    if isinstance(d[k], dict):
                        d[k].update(v)
                    else:
                        d[k] = v
                
                if mn:
                    mn += '-'
                # add "key=value" to model name
                mn += f'{k}={v}'
            
            for i in range(1, self.n_trials+1):
                config2 = dict2AttrDict(copy.deepcopy(config))
                mn2 = copy.copy(mn)

                if self.n_trials > 1:
                    mn2 += f'-trial{i}' if model_name else f'trial{i}'
                if 'seed' in config2['env']:
                    config2['env']['seed'] = 1000 * i
                if 'video_path' in config2['env']:
                    config2['env']['video_path'] = \
                        f'{self.root_dir}/{mn2}/{config2["env"]["video_path"]}'
                
                kw = [f'root_dir={self.root_dir}', f'model_name={mn2}']
                change_config(kw, config2)
                modify_config(config2, root_dir=self.root_dir, model_name=mn2)
                print_dict(config2)
                p = Process(target=self.train_func, args=([config2],))
                p.start()
                self.processes.append(p)
                time.sleep(self.delay)   # ensure sub-processs starts in order
