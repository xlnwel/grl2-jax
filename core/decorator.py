import functools
import logging
import tensorflow as tf

from utility.utils import config_attr
from core.log import setup_logger, setup_tensorboard, save_code, log


logger = logging.getLogger(__name__)


def config(init_fn):
    """ Adds config to attr """
    @functools.wraps(init_fn)
    def wrapper(self, config, *args, name=None, **kwargs):
        config_attr(self, config)
        if name is not None:
            self.name = name

        if 'model_name' in config:
            self._model_name = config['model_name'] or 'baseline'

        init_fn(self, *args, **kwargs)

    return wrapper

def record(init_fn):
    def setup_logger_tensorboard(obj):
        """ Setups Logger and Tensorboard for recording training stats """
        if getattr(obj, '_writer', True):
            obj._writer = setup_tensorboard(obj._root_dir, obj._model_name)
            tf.summary.experimental.set_step(0)

        obj._logger = setup_logger(
            getattr(obj, '_logger', True) and obj._root_dir, obj._model_name)
        for method in dir(obj._logger):
            if not method.startswith('_'):
                setattr(obj, method, getattr(obj._logger, method))
        
        if getattr(obj, '_save_code', True):
            save_code(obj._root_dir, obj._model_name)
        
        obj.log = functools.partial(log, 
            obj._logger, obj._writer, obj._model_name, None)

    @functools.wraps(init_fn)
    def wrapper(self, **kwargs):        
        setup_logger_tensorboard(self)

        init_fn(self, **kwargs)

    return wrapper

def step_track(train_log):
    """ Tracks the training and environment steps """
    @functools.wraps(train_log)
    def wrapper(self, step=0, **kwargs):
        if step > self.env_step:
            self.env_step = step
        n = train_log(self, step, **kwargs)
        self.train_step += n
        return self.train_step

    return wrapper

def override(cls):
    @functools.wraps(cls)
    def check_override(method):
        if method.__name__ not in dir(cls):
            raise NameError("{} does not override any method of {}".format(
                method, cls))
        return method

    return check_override
