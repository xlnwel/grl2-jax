import functools
import logging
import tensorflow as tf

from utility.utils import config_attr
from core.record import *


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

def setup_tensorboard(init_fn):
    def _setup(obj):
        if getattr(obj, '_writer', True):
            obj._writer = creater_tensorboard_writer(obj._root_dir, obj._model_name)
            tf.summary.experimental.set_step(0)
        
        if hasattr(obj, 'record'):
            obj.record = functools.partial(obj.record, recorder=obj._recorder)
        else:
            obj.record = functools.partial(record, 
                writer=obj._writer, model_name=obj._model_name)
    
    @functools.wraps(init_fn)
    def wrapper(self, **kwargs):
        _setup(self)
        init_fn(self, **kwargs)
    
    return wrapper

def setup_recorder(init_fn):
    def _setup(obj):
        """ Setups Logger and Tensorboard for recording training stats """
        obj._recorder = create_recorder(
            getattr(obj, '_recorder', True) and obj._root_dir, obj._model_name)
        for method in dir(obj._recorder):
            if not method.startswith('_'):
                setattr(obj, method, getattr(obj._recorder, method))
        
        if getattr(obj, '_save_code', True):
            save_code(obj._root_dir, obj._model_name)
        
        if hasattr(obj, 'record'):
            obj.record = functools.partial(obj.record, recorder=obj._recorder)
        else:
            obj.record = functools.partial(record, 
                recorder=obj._recorder, model_name=obj._model_name)

    @functools.wraps(init_fn)
    def wrapper(self, **kwargs):        
        _setup(self)

        init_fn(self, **kwargs)

    return wrapper

def step_track(train_record):
    """ Tracks the training and environment steps """
    @functools.wraps(train_record)
    def wrapper(self, step=0, **kwargs):
        if step > self.env_step:
            self.env_step = step
        n = train_record(self, step, **kwargs)
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
