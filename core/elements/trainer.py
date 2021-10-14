import tensorflow as tf

from core.checkpoint import *
from core.elements.loss import Loss
from core.module import EnsembleWithCheckpoint, constructor
from core.optimizer import create_optimizer
from utility.display import display_model_var_info
from utility.utils import config_attr, eval_config


class Trainer(tf.Module):
    def __init__(self, 
                 *,
                 config: dict,
                 loss: Loss,
                 env_stats,
                 name):
        self._raw_name = name
        super().__init__(name=f'{name}_trainer')
        config = config.copy()
        config_attr(self, config, filter_dict=True)
        
        self.loss = loss
        self.env_stats = env_stats

        modules = tuple(v for k, v in self.loss.model.items() 
            if not k.startswith('target'))
        opt_config = eval_config(config.pop('optimizer'))
        self.optimizer = create_optimizer(modules, opt_config)
        
        self.train = tf.function(self.raw_train)
        self._build_train(env_stats)
        self._has_ckpt = 'root_dir' in config and 'model_name' in config
        if self._has_ckpt:
            self.ckpt, self.ckpt_path, self.ckpt_manager = setup_checkpoint(
                {'optimizer': self.optimizer}, self._root_dir, 
                self._model_name, name=self.name)
            if config.get('display_var', True):
                display_model_var_info(self.loss.model)
        self._post_init(config, env_stats)
        self.loss.model.sync_nets()

    def get_weights(self):
        return self.optimizer.get_weights()

    def set_weights(self, weights):
        self.optimizer.set_weights(weights[f'{self._raw_name}_opt'])

    def ckpt_model(self):
        return {
            f'{self._raw_name}_opt': self.optimizer, 
        }

    def _build_train(self, env_stats):
        pass

    def raw_train(self):
        raise NotImplementedError

    def _post_init(self, config, env_stats):
        """ Add some additional attributes and do some post processing here """
        pass

    """ Save & Restore Optimizer """
    def save(self, print_terminal_info=True):
        if self._has_ckpt:
            save(self.ckpt_manager, print_terminal_info)
        else:
            raise RuntimeError(
                'Cannot perform <save> as root_dir or model_name was not specified at initialization')

    def restore(self):
        if self._has_ckpt:
            restore(self.ckpt_manager, self.ckpt, self.ckpt_path, self.name)
        else:
            raise RuntimeError(
                'Cannot perform <restore> as root_dir or model_name was not specified at initialization')


class TrainerEnsemble(EnsembleWithCheckpoint):
    def __init__(self, 
                 *, 
                 config, 
                 model,
                 loss,
                 env_stats,
                 constructor=constructor, 
                 name, 
                 **classes):
        super().__init__(
            config=config, 
            constructor=constructor, 
            name=name, 
            **classes
        )

        self.model = model
        self.loss = loss
        self.env_stats = env_stats

        self.train = tf.function(self.raw_train)
        self._build_train(env_stats)
        if config.get('display_var', True):
            display_model_var_info(self.components)

    def _build_train(self, env_stats):
        raise NotImplementedError
    
    def raw_train(self):
        raise NotImplementedError