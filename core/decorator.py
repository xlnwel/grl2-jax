from functools import wraps
import functools
import logging
import tensorflow as tf

from utility.utils import config_attr
from core.log import setup_logger, setup_tensorboard, save_code, log


logger = logging.getLogger(__name__)


def _setup_monitor(obj, config):
    if config.get('writer', True):
        obj._writer = setup_tensorboard(obj._root_dir, obj._model_name)
        tf.summary.experimental.set_step(0)

    obj._logger = setup_logger(
        config.get('logger', True) and obj._root_dir, obj._model_name)
    for method in dir(obj._logger):
        if not method.startswith('_'):
            setattr(obj, method, getattr(obj._logger, method))
    
    if config.get('save_code', True):
        save_code(obj._root_dir, obj._model_name)
    
    obj.log = functools.partial(log, obj._logger, obj._writer,
        obj._model_name, None)

def record(init_fn):
    """ Setups Tensorboard for recording """
    def wrapper(self, *, config, name, **kwargs):
        config_attr(self, config)
        self.name = name or config["algorithm"]
        
        _setup_monitor(self, config)

        init_fn(self, **kwargs)

    return wrapper

def config(init_fn):
    """ Adds config to attr """
    def wrapper(self, config, *args, **kwargs):
        config_attr(self, config)
        if 'model_name' in config:
            self._model_name = config['model_name'] or 'baseline'

        init_fn(self, *args, **kwargs)

    return wrapper

def step_track(learn_log):
    """ Tracks the training and environment steps """
    @wraps(learn_log)
    def wrapper(self, step=0, **kwargs):
        if step > self.env_step:
            self.env_step = step
        n = learn_log(self, step, **kwargs)
        self.train_step += n
        return self.train_step

    return wrapper

def override(cls):
    @wraps(cls)
    def check_override(method):
        if method.__name__ not in dir(cls):
            raise NameError("{} does not override any method of {}".format(
                method, cls))
        return method

    return check_override
