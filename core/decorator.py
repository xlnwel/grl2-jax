import tensorflow as tf

from utility.display import display_var_info, pwc
from core.checkpoint import setup_checkpoint
from core.log import setup_logger, setup_tensorboard


""" Functions used to print useful information """                    
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
    from functools import wraps
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
        [setattr(self, k, v) for k, v in config.items()]

        self.logger = setup_logger(self.root_dir, self.model_name)
        self.writer = setup_tensorboard(self.root_dir, self.model_name)

        # track models and optimizers for Checkpoint
        self.ckpt_models = {}
        for name_, model in models.items():
            setattr(self, name_, model)
            if isinstance(model, tf.Module) or isinstance(model, tf.Variable):
                self.ckpt_models[name_] = model

        # define global steps for train/env step tracking
        self.global_steps = tf.Variable(0, dtype=tf.int64)

        # Agent initialization
        init_fn(self, name=self.name, config=config, models=models, **kwargs)

        # postprocessing
        self.ckpt, self.ckpt_path, self.ckpt_manager = \
            setup_checkpoint(self.ckpt_models, self.root_dir, self.model_name, self.global_steps)
        display_model_var_info(self.ckpt_models)
        self.print_construction_complete()
    
    return wrapper
