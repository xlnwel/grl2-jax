from functools import wraps
import logging
import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import global_policy

from utility.display import display_model_var_info
from core.checkpoint import setup_checkpoint
from core.log import setup_logger, setup_tensorboard, save_code
from core.optimizer import Optimizer


logger = logging.getLogger(__name__)

def agent_config(init_fn):
    """ Decorator for agent's initialization """
    def wrapper(self, *, name=None, config, models, env, **kwargs):
        """
        Args:
            name: Agent's name
            config: configuration for agent, 
                should be read from config.yaml
            models: a dict of models
            kwargs: optional arguments for each specific agent
        """
        # preprocessing
        # name is used for better bookkeeping, 
        # while model_name is used for create save/log files
        # e.g., all workers share the same name, but with differnt model_names
        self.name = name or f'{config["algorithm"]}-{env.name}'
        """ For the basic configuration, see config.yaml in algo/*/ """
        _config_attr(self, config)

        self._dtype = global_policy().compute_dtype

        self.model = models
        # track models and optimizers for Checkpoint
        self._ckpt_models = {}
        for name_, model in models.items():
            setattr(self, name_, model)
            if isinstance(model, tf.Module) or isinstance(model, tf.Variable):
                self._ckpt_models[name_] = model
        
        self._env_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._train_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.env_step = 0
        self.train_step = 0

        if config.get('writer', True):
            self._writer = setup_tensorboard(self._root_dir, self._model_name)
            tf.summary.experimental.set_step(0)

        # Agent initialization
        init_fn(self, env=env, **kwargs)

        # save optimizers
        for k, v in vars(self).items():
            if isinstance(v, Optimizer):
                self._ckpt_models[k[1:]] = v
        logger.info(f'ckpt models: {self._ckpt_models}')

        self.print_construction_complete()
        
        if config.get('display_var', True):
            display_model_var_info(self._ckpt_models)

        if config.get('save_code', True):
            save_code(self._root_dir, self._model_name)
        
        self._ckpt, self._ckpt_path, self._ckpt_manager = \
            setup_checkpoint(self._ckpt_models, self._root_dir, 
                            self._model_name, self._env_step, self._train_step)

        self.restore()
        
        # to save stats to files, specify `logger: True` in config.yaml 
        self._logger = setup_logger(
            config.get('logger', None) and self._root_dir, 
            self._model_name)
    
    return wrapper

def config(init_fn):
    def wrapper(self, config, *args, **kwargs):
        _config_attr(self, config)

        init_fn(self, *args, **kwargs)

    return wrapper

def step_track(learn_log):
    @wraps(learn_log)
    def wrapper(self, step):
        if step > self.env_step:
            self.env_step = step
        self.train_step += learn_log(self, step)

    return wrapper

def override(cls):
    @wraps(cls)
    def check_override(method):
        if method.__name__ not in dir(cls):
            raise NameError("{} does not override any method of {}".format(
                method, cls))
        return method

    return check_override

def _config_attr(obj, config):
    for k, v in config.items():
        if not k.isupper():
            k = f'_{k}'
        if isinstance(v, str):
            try:
                v = float(v)
            except:
                pass
        if isinstance(v, float) and v == int(v):
            v = int(v)
        setattr(obj, k, v)
