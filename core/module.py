import inspect
import functools
import tensorflow as tf
from tensorflow.keras import layers

from core.ckpt.tf import *
from utility.typing import AttrDict
from utility.utils import set_path


def constructor(config, env_stats, cls, name):
    return cls(config=config, env_stats=env_stats, name=name)


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

        self._results = []

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
        
        self._results.clear()
        for l in self._layers:
            if isinstance(l, training_cls):
                x = l(x, training=training)
            else:
                x = l(x, **kwargs)
            self._results.append(x)
        return x

    def get_weights(self):
        return [v.numpy() for v in self.variables]

    def set_weights(self, weights):
        if isinstance(weights, str):
            self.initialize(weights)
        else:
            assert len(self.variables) == len(weights), (len(self.variables), len(weights))
            [v.assign(w if v.shape == w.shape else
                tf.pad(w, [(0, v_dim - w_dim) for v_dim, w_dim in zip(v.shape, w.shape)]))
                for v, w in zip(self.variables, weights)]

    def mlp(self, x, *args, name, **kwargs):
        if not hasattr(self, f'_{name}'):
            from nn.func import mlp
            setattr(self, f'_{name}', mlp(*args, name=name, **kwargs))
        return getattr(self, f'_{name}')(x)

    def initialize(self, name):
        from nn.utils import get_initializer
        initializer = get_initializer(name)
        for v in self.variables:
            v.assign(initializer(v.shape))

    def results(self):
        return self._results


class Ensemble(tf.Module):
    def __init__(self, 
                 *, 
                 config: AttrDict, 
                 env_stats: AttrDict=None,
                 constructor=constructor, 
                 name: str, 
                 has_ckpt: bool=True,
                 **classes):
        """ Two ways to construct an Ensemble
        1. with <classes> specified, constructor creates a component
        at a time with a dict from <config>, a class from <classes>,
        and a name from the common keys of <config> and <classes> 
        as its arguments. See method <constructor> for an example
        2. without <classes>, constructor create all components at once
        with <config> as its only argument. See for an example:
        <core.elements.construct_components>
        """
        super().__init__(name=name)

        self.config = config
        self.env_stats = env_stats

        self._pre_init()
        self._init_components(constructor, classes)
        if has_ckpt:
            self._ckpt = TFCheckpoint(self.config, self.ckpt_model(), self.name)
        else:
            self._ckpt = None
        self._post_init()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"Attempted to get missing private attribute '{name}'")
        elif hasattr(self._ckpt, name):
            return getattr(self._ckpt, name)
        raise AttributeError(f"Attempted to get missing attribute '{name}'")

    def _pre_init(self):
        """ Add some additional attributes and 
        do some pre-processing here """
        pass

    def _init_components(self, constructor, classes):
        if classes:
            component_configs = [k for k, v in self.config.items() 
                if isinstance(v, dict)]
            if set(component_configs) != set(classes):
                raise ValueError(
                    f'Inconsistent configs and classes: '
                    f'config({component_configs}) != classes({classes})'
                )
            self.components = {}
            for k, cls in classes.items():
                obj = constructor(self.config[k], self.env_stats, cls, k)
                self.components[k] = obj
                setattr(self, k, obj)
        else:
            self.components = constructor(self.config)
            [setattr(self, n, m) for n, m in self.components.items()]

    def _post_init(self):
        """ Add some additional attributes and 
        do some post processing here """
        pass

    """ Auxiliary functions that make Ensemble like a dict """
    def __getitem__(self, key):
        return self.components[key]

    def __setitem__(self, key, value):
        assert key not in self.components, list(self.components)
        self.components[key] = value

    def __contains__(self, item):
        return item in self.components

    def __len__(self):
        return len(self.components)
    
    def __iter__(self):
        return self.components.__iter__()

    def keys(self):
        return self.components.keys()

    def values(self):
        return self.components.values()
    
    def items(self):
        return self.components.items()

    """ Checkpoint Operations """
    def ckpt_model(self):
        ckpt_models = {f'{k}_{kk}': vv
            for k, v in self.components.items() 
            for kk, vv in v.ckpt_model().items()}
        return ckpt_models

    def reset_model_path(self, model_path: ModelPath):
        self.config = set_path(self.config, model_path, max_layer=0)
        self._ckpt.reset_model_path(model_path)

    def restore(self):
        if self._ckpt:
            self._ckpt.restore()
        else:
            raise RuntimeError(f'Cannot perform <restore> as {self.name} has not setup checkpoint')

    def save(self, print_terminal_info=False):
        if self._ckpt:
            self._ckpt.save(print_terminal_info)
        else:
            raise RuntimeError(f'Cannot perform <save> as {self.name} has not setup checkpoint')
