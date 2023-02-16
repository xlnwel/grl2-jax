import jax
import haiku as hk
from typing import Dict, Union

from core.log import do_logging
from core.ckpt.base import ParamsCheckpointBase
from core.ensemble import Ensemble, constructor
from core.typing import AttrDict, dict2AttrDict
from nn.func import create_network


class Model(ParamsCheckpointBase):
    """ A model, consisting of multiple modules, is a 
    self-contained unit for network inference. Its 
    subclass is expected to implement some methods 
    of practical meaning, such as <action> and 
    <compute_value> """
    def __init__(
        self, 
        *,
        config: AttrDict,
        env_stats: AttrDict,
        name: str,
    ):
        super().__init__(config, name)
        self.env_stats = dict2AttrDict(env_stats, to_copy=True)
        self.modules: Dict[str, hk.Module] = AttrDict()
        self.rng = self._prngkey()
        self.act_rng = self.rng

        self.add_attributes()
        self.build_nets()
        self.compile_model()

    def _prngkey(self, seed=None):
        if seed is None:
            if self.config.seed is None:
                self.config.seed = 42
            seed = self.config.seed
        do_logging(f'Model({self.name}) seed: {seed}')
        return jax.random.PRNGKey(seed)

    def add_attributes(self):
        pass

    def build_net(self, *args, name, return_init=False, **kwargs):
        def build(*args, **kwargs):
            net = create_network(self.config[name], name)
            return net(*args, **kwargs)
        net = hk.transform(build)
        if return_init:
            return net.init, net.apply
        else:
            self.rng, rng = jax.random.split(self.rng)
            self.act_rng = self.rng
            return net.init(rng, *args, **kwargs), net.apply

    def build_nets(self):
        raise NotImplementedError

    def compile_model(self):
        self.jit_action = jax.jit(self.raw_action, static_argnames=('evaluation'))
        # self.jit_action = jax.jit(self.raw_action, static_argnums=(3))

    def action(self, data, evaluation):
        self.act_rng, act_rng = jax.random.split(self.act_rng)
        return self.jit_action(
            self.params, act_rng, data, evaluation)

    def raw_action(self, params, rng, data, evaluation=False):
        raise NotImplementedError

    def get_weights(self, name: str=None):
        """ Returns a list/dict of weights

        Returns:
            If name is provided, it returns a dict of weights 
            for models specified by keys. Otherwise, it 
            returns a list of all weights
        """
        if name is None:
            name = list(self.params.keys())
        elif isinstance(name, str):
            name = [name]
        assert isinstance(name, (tuple, list))

        weights = {n: self.params[n] for n in name}
        return weights

    def set_weights(self, weights: dict):
        """ Sets weights

        Args:
            weights: a dict or list of weights. If it's a dict, 
            it sets weights for models specified by the keys.
            Otherwise, it sets all weights 
        """
        assert set(weights).issubset(set(self.params)) or set(self.params).issubset(set(weights)), (list(self.params), list(weights))
        for name in self.params.keys():
            if name in weights:
                self.params[name] = weights[name]

    def get_states(self):
        pass
    
    def reset_states(self, state=None):
        pass

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return None

    @property
    def state_size(self):
        return None

    @property
    def state_keys(self):
        return None

    @property
    def state_type(self):
        return None


class ModelEnsemble(Ensemble):
    def __init__(
        self, 
        *, 
        config: dict, 
        env_stats: dict,
        constructor=constructor, 
        components=None, 
        name: str, 
        to_build=False, 
        to_build_for_eval=False,
        **classes
    ):
        super().__init__(
            config=config, 
            env_stats=env_stats, 
            constructor=constructor, 
            components=components, 
            name=name, 
            has_ckpt=False, 
            **classes
        )

    def get_weights(self, name: Union[dict, list]=None):
        weights = {}
        if name:
            if isinstance(name, dict):
                for model_name, comp_name in name.items():
                    weights[model_name] = self.components[model_name].get_weights(comp_name)
            elif isinstance(name, list):
                for model_name in name:
                    weights[model_name] = self.components[model_name].get_weights()
        else:
            for k, v in self.components.items():
                weights[k] = v.get_weights()
        return weights

    def set_weights(self, weights: Union[list, dict], default_initialization=None):
        for n, m in self.components.items():
            if n in weights:
                m.set_weights(weights[n], default_initialization)
            elif default_initialization:
                m.set_weights({}, default_initialization)

    def restore(self):
        for v in self.components.values():
            v.restore()

    def save(self):
        for v in self.components.values():
            v.save()
