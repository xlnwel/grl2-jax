from typing import Union

from core.checkpoint import *
from core.module import Ensemble, EnsembleWithCheckpoint


def construct_components(config):
    from nn.func import create_network
    return {k: create_network(v, name=k) 
        for k, v in config.items() if isinstance(v, dict)}


class Model(Ensemble):
    """ A model, consisting of multiple modules, is a 
    self-contained unit for network inference. Its 
    subclass is expected to implement some methods 
    of practical meaning, such as <action> and 
    <compute_value> """
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


class ModelEnsemble(EnsembleWithCheckpoint):
    pass
