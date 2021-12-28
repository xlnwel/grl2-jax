from typing import Union

from core.checkpoint import *
from core.module import Ensemble, constructor
from utility.typing import AttrDict


def construct_components(config):
    from nn.func import create_network
    networks = {k: create_network(v, name=k) 
        for k, v in config.items() if isinstance(v, dict)}
    return networks


class Model(Ensemble):
    """ A model, consisting of multiple modules, is a 
    self-contained unit for network inference. Its 
    subclass is expected to implement some methods 
    of practical meaning, such as <action> and 
    <compute_value> """
    def __init__(self, 
                 *,
                 config: AttrDict,
                 env_stats: AttrDict=None,
                 constructor=construct_components,
                 name: str,
                 to_build=False,
                 to_build_for_eval=False):
        assert env_stats is not None, env_stats
        self._pre_init(config, env_stats)
        super().__init__(
            config=config, 
            env_stats=env_stats,
            constructor=constructor, 
            name=name)

        self._has_ckpt = 'root_dir' in config and 'model_name' in config
        if to_build:
            self._build(env_stats)
            self.to_build = True
        elif to_build_for_eval:
            self._build(env_stats, evaluation=True)
            self.to_build = True
        else:
            self.to_build = False
        self._post_init()

    def ckpt_model(self):
        return self.components

    def _pre_init(self, config, env_stats):
        pass

    def _build(self, env_stats, evaluation=False):
        pass

    def _post_init(self):
        """ Add some additional attributes and 
        do some post processing here """
        pass

    def sync_nets(self):
        """ Sync target network """
        if hasattr(self, '_sync_nets'):
            # defined in TargetNetOps
            self._sync_nets()

    def get_weights(self, name: str=None):
        """ Returns a list/dict of weights

        Returns:
            If name is provided, it returns a dict of weights 
            for models specified by keys. Otherwise, it 
            returns a list of all weights
        """
        if name is None:
            return [v.numpy() for v in self.variables]
        elif isinstance(name, str):
            name = [name]
        assert isinstance(name, (tuple, list))

        return {n: self.components[n].get_weights() for n in name}

    def set_weights(self, weights: Union[list, dict], default_initialization=None):
        """ Sets weights

        Args:
            weights: a dict or list of weights. If it's a dict, 
            it sets weights for models specified by the keys.
            Otherwise, it sets all weights 
        """
        if isinstance(weights, dict):
            for n, m in self.components.items():
                if n in weights:
                    m.set_weights(weights[n])
                elif default_initialization:
                    m.set_weights(default_initialization)
        else:
            if len(weights) == 0:
                return
            assert len(self.variables) == len(weights), \
                (len(self.variables), len(weights))
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


class ModelEnsemble(Ensemble):
    def __init__(self, 
                 *, 
                 config: dict, 
                 env_stats: dict,
                 constructor=constructor, 
                 name: str, 
                 to_build=False, 
                 to_build_for_eval=False,
                 **classes):
        super().__init__(
            config=config, 
            env_stats=env_stats, 
            constructor=constructor, 
            name=name, 
            **classes
        )
        if to_build:
            self._build(env_stats)
            self.to_build = True
        elif to_build_for_eval:
            self._build(env_stats, evaluation=True)
            self.to_build = True
        else:
            self.to_build = False

    def get_weights(self, name: Union[dict, list]=None):
        if name:
            weights = {}
            if isinstance(name, dict):
                for model_name, comp_name in name.items():
                    weights[model_name] = self.components[model_name].get_weights(comp_name)
            elif isinstance(name, list):
                for model_name in name:
                    weights[model_name] = self.components[model_name].get_weights()
            return weights
        else:
            return [v.numpy() for v in self.variables]

    def set_weights(self, weights: Union[list, dict], default_initialization=None):
        if isinstance(weights, dict):
            for n, m in self.components.items():
                if n in weights:
                    m.set_weights(weights[n], default_initialization)
                elif default_initialization:
                    m.set_weights({}, default_initialization)
        else:
            if len(weights) == 0:
                return
            assert len(self.variables) == len(weights), \
                (len(self.variables), len(weights))
            [v.assign(w) for v, w in zip(self.variables, weights)]
