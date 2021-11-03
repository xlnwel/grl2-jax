import tensorflow as tf

from core.checkpoint import *
from core.elements.loss import Loss, LossEnsemble
from core.module import EnsembleWithCheckpoint, constructor
from core.optimizer import create_optimizer
from utility.display import display_model_var_info
from utility.utils import config_attr, eval_config


class Trainer(tf.Module):
    def __init__(self, 
                 *,
                 config: dict,
                 loss: Loss,
                 env_stats: dict,
                 name: str):
        self._raw_name = name
        super().__init__(name=f'{name}_trainer')
        config = config.copy()
        config_attr(self, config, filter_dict=True)
        
        self.model = loss.model
        self.loss = loss
        self.env_stats = env_stats

        modules = tuple(v for k, v in self.model.items() 
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
                display_model_var_info(self.model)
        self._post_init(config, env_stats)
        self.model.sync_nets()

    def get_weights(self, identifier=None):
        if identifier is None:
            identifier = self._raw_name
        weights = {
            f'{identifier}_model': self.model.get_weights(),
            f'{identifier}_opt': self.get_optimizer_weights(),
        }
        return weights

    def set_weights(self, weights, identifier=None):
        if identifier is None:
            identifier = self._raw_name
        self.model.set_weights(weights[f'{identifier}_model'])
        self.set_optimizer_weights(weights[f'{identifier}_opt'])
        
    def get_model_weights(self, name: str=None):
        return self.model.get_weights(name)

    def set_model_weights(self, weights):
        self.model.set_weights(weights)

    def get_optimizer_weights(self):
        return self.optimizer.get_weights()

    def set_optimizer_weights(self, weights):
        self.optimizer.set_weights(weights)

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
    def save_optimizer(self, print_terminal_info=False):
        if self._has_ckpt:
            save(self.ckpt_manager, print_terminal_info)
        else:
            raise RuntimeError(
                'Cannot perform <save> as root_dir or model_name was not specified at initialization')

    def restore_optimizer(self):
        if self._has_ckpt:
            restore(self.ckpt_manager, self.ckpt, self.ckpt_path, self.name)
        else:
            raise RuntimeError(
                'Cannot perform <restore> as root_dir or model_name was not specified at initialization')

    def save(self, print_terminal_info=False):
        self.save_optimizer(print_terminal_info)
        self.model.save(print_terminal_info)
    
    def restore(self):
        self.restore_optimizer()
        self.model.restore()


class TrainerEnsemble(EnsembleWithCheckpoint):
    def __init__(self, 
                 *, 
                 config, 
                 loss: LossEnsemble,
                 env_stats: dict,
                 constructor=constructor, 
                 name: str, 
                 **classes):
        super().__init__(
            config=config, 
            constructor=constructor, 
            name=name, 
            **classes
        )

        self.model = loss.model
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

    def restore(self):
        super().restore()
        self.model.restore()

    def save(self, print_terminal_info=False):
        super().save(print_terminal_info)
        self.model.save(print_terminal_info)

    def restore_optimizer(self):
        super().restore()

    def save_optimizer(self, print_terminal_info=False):
        super().save(print_terminal_info)


def create_trainer(config, loss, env_stats, *, name, trainer_cls, **kwargs):
    trainer = trainer_cls(
        config=config, loss=loss, 
        env_stats=env_stats, name=name, **kwargs)

    return trainer
