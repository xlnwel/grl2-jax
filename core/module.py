import inspect
import functools
from typing import Union
from utility.display import display_model_var_info
import tensorflow as tf
from tensorflow.keras import layers

from core.checkpoint import *
from core.optimizer import create_optimizer
from utility.utils import config_attr, eval_config


def construct_components(config):
    from nn.func import create_network
    return {k: create_network(v, name=k) 
        for k, v in config.items() if isinstance(v, dict)}


class Module(tf.Module):
    """ This class is an alternative to keras.layers.Layer when 
    encapsulating multiple layers. It provides more fine-grained 
    output for keras.Model.summary and automatically handles 
    training signal for batch normalization and dropout. 
    Moreover, you can now, without worry about name conflicts, 
    define `self._layers`, which is by default used in <call>.
    """
    def __init__(self, name):
        self.scope_name = name
        self._is_built = False
        self._training_cls = [layers.BatchNormalization, layers.Dropout]
        name = name and name.split('/')[-1]
        super().__init__(name=name)

    def __call__(self, *args, **kwargs):
        if not self._is_built:
            self._build(*tf.nest.map_structure(
                lambda x: x.shape if hasattr(x, 'shape') else x, args))
        if hasattr(self, '_layers') and isinstance(self._layers, (list, tuple)):
            self._layers = [l for l in self._layers if isinstance(l, tf.Module)]

        return self._call(*args, **kwargs)
        
    def _build(self, *args):
        self.build(*args)
        self._is_built = True

    def build(self, *args, **kwargs):
        """ Override this if necessary """
        pass

    # @tf.Module.with_name_scope    # do not decorate with this as it will introduce inconsistent variable names between keras.Model and plain call
    def _call(self, *args, **kwargs):
        return self.call(*args, **kwargs)
        
    def call(self, x, training=False, training_cls=(), **kwargs):
        """ Override this if necessary """
        training_cls = set(training_cls) | set(self._training_cls)
        training_cls = tuple([c.func if isinstance(c, functools.partial) 
            else c for c in training_cls if inspect.isclass(c)])
        
        for l in self._layers:
            if isinstance(l, training_cls):
                x = l(x, training=training)
            else:
                x = l(x, **kwargs)
        return x

    def get_weights(self):
        return [v.numpy() for v in self.variables]

    def set_weights(self, weights):
        [v.assign(w) for v, w in zip(self.variables, weights)]

    def mlp(self, x, *args, name, **kwargs):
        if not hasattr(self, f'_{name}'):
            from nn.func import mlp
            setattr(self, f'_{name}', mlp(*args, name=name, **kwargs))
        return getattr(self, f'_{name}')(x)


class Model(tf.Module):
    """ A model, consisting of multiple modules, is a 
    self-contained unit for inference. Its subclass is 
    expected to implement some methods of practical 
    meaning, such as <action> and <compute_value> """
    def __init__(self, 
                 *,
                 config,
                 model_fn=construct_components,
                 name):
        super().__init__(f'{name}_model')
        config = config.copy()
        config_attr(self, config, filter_dict=True)

        self.modules = model_fn(config)
        self.ckpt, self.ckpt_path, self.ckpt_manager = \
            setup_checkpoint(self.modules, self._root_dir, 
                self._model_name, name=self.name)

        # add modules as public attributes
        [setattr(self, n, m) for n, m in self.modules.items()]
        self._post_init(config)

    def _post_init(self):
        """ Add some additional attributes and do some post processing here """
        pass

    def sync_nets(self):
        """ Sync target network """
        if hasattr(self, '_sync_nets'):
            # defined in TargetNetOps
            self._sync_nets()

    def get_weights(self, name: str=None):
        """ Returns a list/dict of weights

        Returns:
            If name is provided, it returns a dict of weights for models 
            specified by keys. Otherwise it returns a list of all weights
        """
        if name is None:
            return [v.numpy() for v in self.variables]
        elif isinstance(name, str):
            name = [name]
        assert isinstance(name, (tuple, list))

        return {n: self.modules[n].get_weights() for n in name}

    def set_weights(self, weights: Union[list, dict]):
        """ Sets weights

        Args:
            weights: a dict or list of weights. If it's a dict, 
            it sets weights for models specified by the keys.
            Otherwise, it sets all weights 
        """
        if isinstance(weights, dict):
            for n, w in weights.items():
                self[n].set_weights(w)
        else:
            assert len(self.variables) == len(weights), \
                (len(self.variables), len(weights), weights)
            [v.assign(w) for v, w in zip(self.variables, weights)]
    
    def reset_states(self, states=None):
        pass

    def get_states(self):
        pass

    @property
    def state_size(self):
        return self.rnn.state_size if hasattr(self, 'rnn') else None

    @property
    def state_keys(self):
        return self.rnn.state_keys if hasattr(self, 'rnn') else ()

    """ Auxiliary functions that make Model like a dict """
    def __getitem__(self, key):
        return self.modules[key]

    def __setitem__(self, key, value):
        assert key not in self.modules, list(self.modules)
        self.modules[key] = value
    
    def __contains__(self, item):
        return item in self.modules

    def __len__(self):
        return len(self.modules)
    
    def __iter__(self):
        return self.modules.__iter__()

    def keys(self):
        return self.modules.keys()

    def values(self):
        return self.modules.values()
    
    def items(self):
        return self.modules.items()

    """ Save & Restore Model """
    def save(self, print_terminal_info=True):
        save(self.ckpt_manager, print_terminal_info)
    
    def restore(self):
        restore(self.ckpt_manager, self.ckpt, self.ckpt_path, self.name)


class Loss:
    """ We do not need this class for now. However, 
    in case one day we need multiple GPUs for training, 
    this class will come in handy.
    """
    def __init__(self,
                 *,
                 config: dict,
                 model: Model,
                 use_tf: bool=False):
        config = config.copy()
        config_attr(self, config, filter_dict=True)

        self.model = model
        [setattr(self, k, v) for k, v in self.model.items()]
        if use_tf:
            self.loss = tf.function(self._loss)

    def _loss(self):
        raise NotImplementedError


class Trainer:
    def __init__(self, 
                 *,
                 config: dict,
                 model: Model, 
                 loss: Loss,
                 env_stats,
                 name):
        self.name = f'{name}_opt'
        config = config.copy()
        config_attr(self, config, filter_dict=True)
        
        self.model = model
        self.loss = loss
        modules = [v for k, v in self.model.items() 
            if not k.startswith('target')]
        opt_config = eval_config(config.pop('optimizer'))
        self.optimizer = create_optimizer(modules, opt_config)

        self.ckpt, self.ckpt_path, self.ckpt_manager = setup_checkpoint(
            {'optimizer': self.optimizer}, self._root_dir, 
            self._model_name, name=self.name)
        
        self._build_learn(env_stats)
        if config.get('display_var', True):
            display_model_var_info(self.model)
        self._post_init()
        self.model.sync_nets()

    # def __getattr__(self, name):
    #     if name.startswith('_'):
    #         raise AttributeError(f"attempted to get missing private attribute '{name}'")
    #     return getattr(self.model, name)

    def _build_learn(self, env):
        raise NotImplementedError

    def _learn(self):
        raise NotImplementedError
    
    def _post_init(self):
        """ Add some additional attributes and do some post processing here """
        pass

    """ Save & Restore Model """
    def save(self, print_terminal_info=True):
        self.model.save(print_terminal_info)
        save(self.ckpt_manager, print_terminal_info)
    
    def restore(self):
        self.model.restore()
        restore(self.ckpt_manager, self.ckpt, self.ckpt_path, self.name)
