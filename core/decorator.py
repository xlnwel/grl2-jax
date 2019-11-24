from utility.display import display_var_info, pwc
from core.checkpoint import setup_checkpoint
from core.log import setup_logger, setup_tensorboard


""" Functions for print useful information """                    
def display_model_var_info(models):
    tvars = []
    for name, model in models.items():
        if 'opt' in name or 'target' in name:
            pass # ignore variables in the optimizer and target networks
        else:
            tvars += model.trainable_variables
        
    display_var_info(tvars)

def print_construction_complete(name):
    pwc(f'{name} has been constructed', color='cyan')


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
        self.name = name
        """ For the basic configuration, see config.yaml in algo/*/ """
        [setattr(self, k, v) for k, v in config.items()]

        self.logger = setup_logger(self.log_root_dir, self.model_name)
        self.writer = setup_tensorboard(self.log_root_dir, self.model_name)

        # track models and optimizers for Checkpoint
        self.ckpt_models = models
        for name, model in models.items():
            setattr(self, name, model)

        # Agent initialization
        init_fn(self, name=name, config=config, models=models, **kwargs)
        
        # postprocessing
        self.global_steps, self.ckpt, self.ckpt_path, self.ckpt_manager = \
            setup_checkpoint(self.ckpt_models, self.model_root_dir, self.model_name)
        display_model_var_info(self.ckpt_models)
        print_construction_complete(name)
    
    return wrapper
