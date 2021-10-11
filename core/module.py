import inspect
import functools
from typing import Dict, Tuple, Union
from utility.display import display_model_var_info
import tensorflow as tf
from tensorflow.keras import layers

from core.checkpoint import *
from core.optimizer import create_optimizer
from utility.utils import config_attr, eval_config
from utility.tf_utils import numpy2tensor, tensor2numpy


def construct_components(config):
    from nn.func import create_network
    return {k: create_network(v, name=k) 
        for k, v in config.items() if isinstance(v, dict)}


def constructor(config, cls, name):
    return cls(config=config, name=name)


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


class Ensemble(tf.Module):
    def __init__(self, *, config, constructor=constructor, name, **classes):
        """ Two ways to construct an Ensemble
        1. with <classes> specified, constructor creates a component
        at a time with a dict from <config>, a class from <classes>,
        and a name from the common keys of <config> and <classes> 
        as its arguments. See method <constructor> for an example
        2. without <classes>, constructor create all components at once
        with <config> as its only argument. See <construct_components>
        for an example
        """
        super().__init__(name=name)
        config = config.copy()
        config_attr(self, config, filter_dict=True)

        if classes:
            component_configs = [k for k, v in config.items() 
                if isinstance(v, dict)]
            if set(component_configs) != set(classes):
                raise ValueError(
                    f'Inconsistent configs and classes: '
                    f'config({component_configs}) != classes({classes})'
                )
            self.components = {}
            for k, cls in classes.items():
                obj = constructor(config[k], cls, k)
                self.components[k] = obj
                setattr(self, k, obj)
        else:
            self.components = constructor(config)
            [setattr(self, n, m) for n, m in self.components.items()]

        self._post_init(config)
    
    def _post_init(self, config):
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


class Model(Ensemble):
    """ A model, consisting of multiple modules, is a 
    self-contained unit for inference. Its subclass is 
    expected to implement some methods of practical 
    meaning, such as <action> and <compute_value> """
    def __init__(self, 
                 *,
                 config,
                 model_fn=construct_components,
                 name):
        super().__init__(config=config, 
            constructor=model_fn, name=name)

        self._has_ckpt = 'root_dir' in config and 'model_name' in config

        self._post_init(config)

    def ckpt_model(self):
        return self.components

    def _post_init(self, config):
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

    def get_states(self):
        pass
    
    def reset_states(self, state=None):
        if hasattr(self, 'rnn'):
            self.rnn.reset_states(state)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if hasattr(self, 'rnn'):
            state = self.rnn.get_initial_state(
                inputs, batch_size=batch_size, dtype=dtype)
        else:
            state = None
        return state

    @property
    def state_size(self):
        return self.rnn.state_size if hasattr(self, 'rnn') else None

    @property
    def state_keys(self):
        return self.rnn.state_keys if hasattr(self, 'rnn') else ()

    @property
    def state_type(self):
        return self.rnn.state_type if hasattr(self, 'rnn') else None

    """ Save & Restore Model """
    def setup_checkpoint(self):
        if not hasattr(self, 'ckpt'):
            self.ckpt, self.ckpt_path, self.ckpt_manager = \
                setup_checkpoint(self.components, self._root_dir, 
                    self._model_name, name=self.name)

    def save(self, print_terminal_info=True):
        if self._has_ckpt:
            self.setup_checkpoint()
            save(self.ckpt_manager, print_terminal_info)
        else:
            raise RuntimeError(
                'Cannot perform <save> as root_dir or model_name was not specified at initialization')

    def restore(self):
        if self._has_ckpt:
            self.setup_checkpoint()
            restore(self.ckpt_manager, self.ckpt, self.ckpt_path, self.name)
        else:
            raise RuntimeError(
                'Cannot perform <restore> as root_dir or model_name was not specified at initialization')


class Loss(tf.Module):
    """ We do not need this class for now. However, 
    in case one day we need multiple GPUs for training, 
    this class will come in handy.
    """
    def __init__(self,
                 *,
                 config: dict,
                 model: Model,
                 use_tf: bool=False,
                 name):
        super().__init__(name=f'{name}_loss')
        config = config.copy()
        config_attr(self, config, filter_dict=True)

        self.model = model
        [setattr(self, k, v) for k, v in self.model.items()]
        if use_tf:
            self.loss = tf.function(self.loss)

    def loss(self):
        raise NotImplementedError


class Trainer(tf.Module):
    def __init__(self, 
                 *,
                 config: dict,
                 model: Model, 
                 loss: Loss,
                 env_stats,
                 name):
        self._raw_name = name
        super().__init__(name=f'{name}_trainer')
        config = config.copy()
        config_attr(self, config, filter_dict=True)
        
        self.model = model
        self.loss = loss
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

    def get_weights(self):
        return {
            f'{self._raw_name}_model': self.model.get_weights(),
            f'{self._raw_name}_opt': self.optimizer.get_weights(),
        }

    def set_weights(self, weights):
        self.model.set_weights(weights[f'{self._raw_name}_model'])
        self.optimizer.set_weights(weights[f'{self._raw_name}_opt'])

    def ckpt_model(self):
        return {
            f'{self._raw_name}_model': self.model,
            f'{self._raw_name}_opt': self.optimizer, 
        }

    def _build_train(self, env_stats):
        pass

    def raw_train(self):
        raise NotImplementedError
    
    def _post_init(self, config, env_stats):
        """ Add some additional attributes and do some post processing here """
        pass

    """ Save & Restore Model """
    def save(self, print_terminal_info=True):
        self.model.save(print_terminal_info)
        if self._has_ckpt:
            save(self.ckpt_manager, print_terminal_info)
        else:
            raise RuntimeError(
                'Cannot perform <save> as root_dir or model_name was not specified at initialization')

    def restore(self):
        self.model.restore()
        if self._has_ckpt:
            restore(self.ckpt_manager, self.ckpt, self.ckpt_path, self.name)
        else:
            raise RuntimeError(
                'Cannot perform <restore> as root_dir or model_name was not specified at initialization')


class EnsembleWithCheckpoint(Ensemble):
    def __init__(self, *, config, constructor=constructor, name, **classes):
        super().__init__(
            config=config, 
            constructor=constructor, 
            name=name, 
            **classes
        )

    """ Save & Restore Model """
    def setup_checkpoint(self):
        if not hasattr(self, 'ckpt'):
            ckpt_models = {}
            for v in self.components.values():
                ckpt_models.update(v.ckpt_model())
            print(self.name, 'checkpoint:', ckpt_models)
            self.ckpt, self.ckpt_path, self.ckpt_manager = setup_checkpoint(
                ckpt_models, self._root_dir, self._model_name, name=self.name)
    
    def save(self, print_terminal_info=True):
        self.setup_checkpoint()
        save(self.ckpt_manager, print_terminal_info)
    
    def restore(self):
        self.setup_checkpoint()
        restore(self.ckpt_manager, self.ckpt, self.ckpt_path, self.name)


class ModelEnsemble(EnsembleWithCheckpoint):
    pass


class LossEnsemble(Ensemble):
    pass


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

        self.train = tf.function(self.raw_train)
        self._build_train(env_stats)
        if config.get('display_var', True):
            display_model_var_info(self.components)

    def _build_train(self, env_stats):
        raise NotImplementedError
    
    def raw_train(self):
        raise NotImplementedError


class Actor:
    def __init__(self, *, config, model, name):
        self._raw_name = name
        self._name = f'{name}_actor'
        config = config.copy()
        config_attr(self, config, filter_dict=True)
        
        self.model = model
        
        self._post_init(config)

    @property
    def raw_name(self):
        return self._raw_name

    @property
    def name(self):
        return self._name

    def _post_init(self):
        pass

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.model, name)

    def __call__(self, 
                 inp: dict,  
                 evaluation: bool=False, 
                 return_eval_stats: bool=False):
        """ The interface to interact with the environment
        
        Args:
            inp: input to the calling method
            evaluation: evaluation mode or not
            return_eval_stats: if return evaluation stats
        Return:
            (action, terms, rnn_state)
        """
        inp, tf_inp = self._process_input(inp, evaluation)
        out = self.model.action(
            **tf_inp, 
            evaluation=evaluation,
            return_eval_stats=return_eval_stats)
        out = self._process_output(inp, out, evaluation)

        return out

    """ Overwrite the following methods if necessary """
    def _process_input(self, inp: dict, evaluation: bool):
        """ Processes input to Model at the algorithmic level 
        
        Args:
            inp: input to the model
            evaluation bool: evaluation mode or not
        Returns: 
            processed input to <model.action>
        """
        return inp, numpy2tensor(inp)

    def _process_output(self, 
                        inp: dict, 
                        out: Tuple[tf.Tensor, Dict[str, tf.Tensor]], 
                        evaluation: bool):
        """ Post-processes output. By default, 
        we convert tf.Tensor to np.ndarray
        
        Args:
            inp: Pre-processed inputs
            out: Model output
        Returns:
            (action, terms, rnn_state)
        """
        out = (*tensor2numpy(out[:2]), out[-1])
        return out

    def get_auxiliary_stats(self):
        pass
    
    def set_auxiliary_stats(self):
        pass

    def save_auxiliary_stats(self):
        pass
    
    def restore_auxiliary_stats(self):
        pass
