import tensorflow as tf

from core.checkpoint import *
from core.elements.loss import Loss, LossEnsemble
from core.module import constructor, Ensemble
from core.optimizer import create_optimizer
from utility.display import display_model_var_info
from utility.typing import AttrDict
from utility.utils import set_path


class Trainer(tf.Module):
    def __init__(
        self, 
        *,
        config: AttrDict,
        env_stats: AttrDict,
        loss: Loss,
        name: str
    ):
        self._raw_name = name
        super().__init__(name=f'{name}_trainer')
        self.config = config

        self.model = loss.model
        self.loss = loss
        self.env_stats = env_stats

        # keep the order fixed, otherwise you may encounter 
        # the permutation misalignment problem when restoring from a checkpoint
        keys = sorted([k for k in self.model.keys() if not k.startswith('target')])
        modules = tuple(self.model[k] for k in keys)
        self.optimizer = create_optimizer(modules, self.config.optimizer)
        
        self.train = tf.function(self.raw_train)
        has_built = self._build_train(env_stats)

        self._opt_ckpt = ckpt(self.config, self.ckpt_model(), self.name)

        if has_built and self.config.get('display_var', True):
            display_model_var_info(self.model)
        self._post_init()
        self.model.sync_nets()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"Attempted to get missing private attribute '{name}'")
        elif hasattr(self._opt_ckpt, name):
            return getattr(self._opt_ckpt, name)
        raise AttributeError(f"Attempted to get missing attribute '{name}'")

    def _build_train(self, env_stats):
        return False

    def raw_train(self):
        raise NotImplementedError

    def _post_init(self):
        """ Add some additional attributes and do some post processing here """
        pass

    """ Weights Access """
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

    """ Checkpoints """
    def reset_model_path(self, model_path: ModelPath):
        self.config = set_path(self.config, model_path, max_layer=0)
        self._opt_ckpt.reset_model_path(model_path)

    def save_optimizer(self, print_terminal_info=False):
        self._opt_ckpt.save()

    def restore_optimizer(self):
        self._opt_ckpt.restore()

    def save(self, print_terminal_info=False):
        self.save_optimizer(print_terminal_info)
        self.model.save(print_terminal_info)
    
    def restore(self):
        self.restore_optimizer()
        self.model.restore()


class TrainerEnsemble(Ensemble):
    def __init__(
        self, 
        *, 
        config: dict, 
        env_stats: dict,
        loss: LossEnsemble,
        constructor=constructor, 
        name: str, 
        **classes
    ):
        super().__init__(
            config=config, 
            env_stats=env_stats, 
            constructor=constructor, 
            name=f'{name}_trainer', 
            **classes
        )

        self.model = loss.model
        self.loss = loss

        self.train = tf.function(self.raw_train)
        has_built = self._build_train(env_stats)
        if has_built and config.get('display_var', True):
            display_model_var_info(self.components)

    def _build_train(self, env_stats):
        raise NotImplementedError
    
    def raw_train(self):
        raise NotImplementedError

    """ Checkpoints """
    def get_weights(self):
        weights = {
            f'model': self.model.get_weights(),
            f'opt': self.get_optimizer_weights(),
        }
        return weights

    def set_weights(self, weights):
        self.model.set_weights(weights['model'])
        self.set_optimizer_weights(weights['opt'])

    def get_model_weights(self, name: str=None):
        return self.model.get_weights(name)

    def set_model_weights(self, weights):
        self.model.set_weights(weights)

    def get_optimizer_weights(self):
        return [trainer.get_optimizer_weights() for trainer in self.components.values()]

    def set_optimizer_weights(self, weights):
        [trainer.set_optimizer_weights(w) 
            for trainer, w in zip(self.components.values(), weights)]

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


def create_trainer(config, env_stats, loss, *, name, trainer_cls, **kwargs):
    trainer = trainer_cls(
        config=config, 
        env_stats=env_stats, 
        loss=loss, 
        name=name, 
        **kwargs)

    return trainer
