from functools import wraps
import tensorflow as tf

from utility.display import display_var_info, pwc
from core.checkpoint import setup_checkpoint
from core.log import setup_logger, setup_tensorboard, save_code


""" Functions used to print model variables """                    
def display_model_var_info(models):
    displayed_models = []
    tvars = []
    for name, model in models.items():
        if 'opt' in name or 'target' in name or model in displayed_models:
            pass # ignore variables in the optimizer and target networks
        else:
            displayed_models.append(model)
            tvars += model.trainable_variables
        
    display_var_info(tvars)

def agent_config(init_fn):
    """ Decorator for agent's initialization """
    @wraps(init_fn)
    def wrapper(self, *, name, config, models, **kwargs):
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
        self.name = name
        """ For the basic configuration, see config.yaml in algo/*/ """
        [setattr(self, k if k.isupper() else f'_{k}', v) for k, v in config.items()]

        # track models and optimizers for Checkpoint
        self._ckpt_models = {}
        for name_, model in models.items():
            setattr(self, name_, model)
            if isinstance(model, tf.Module) or isinstance(model, tf.Variable):
                self._ckpt_models[name_] = model
                
        # Agent initialization
        init_fn(self, **kwargs)

        self._logger = setup_logger(self._root_dir, self._model_name)
        self._writer = setup_tensorboard(self._root_dir, self._model_name)

        # define global steps for train/env step tracking
        self.global_steps = tf.Variable(0, dtype=tf.int64)

        if getattr(self, '_save_code', False):
            save_code(self._root_dir, self._model_name)
        
        # postprocessing
        self._ckpt, self._ckpt_path, self._ckpt_manager = \
            setup_checkpoint(self._ckpt_models, self._root_dir, self._model_name, self.global_steps)
        display_model_var_info(self._ckpt_models)
        self.print_construction_complete()
    
    return wrapper

def config(init_fn):
    @wraps(init_fn)
    def wrapper(self, config, *args, **kwargs):
        [setattr(self, k if k.isupper() else f'_{k}', v) for k, v in config.items()]

        init_fn(self, *args, **kwargs)

    return wrapper