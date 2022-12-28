import logging
from typing import Dict, Sequence, Union

from .typing import ModelPath
from core.typing import AttrDict, dict2AttrDict
from tools.utils import set_path, add_attr

logger = logging.getLogger(__name__)

def constructor(config, env_stats, cls, name):
    return cls(config=config, env_stats=env_stats, name=name)


class Ensemble:
    def __init__(
        self, 
        *, 
        config: AttrDict, 
        env_stats: AttrDict=None,
        components: Dict=None, 
        name: str, 
    ):
        """ Two ways to construct an Ensemble
        1. with <classes> specified, constructor creates a component
        at a time with a dict from <config>, a class from <classes>,
        and a name from the common keys of <config> and <classes> 
        as its arguments. See method <constructor> for an example
        2. without <classes>, constructor create all components at once
        with <config> as its only argument. See for an example:
        <core.elements.construct_components>
        """
        self.name = name
        self.config = dict2AttrDict(config, to_copy=True)
        if env_stats is not None:
            self.env_stats = dict2AttrDict(env_stats, to_copy=True)
        else:
            self.env_stats = None
        self.prev_init()
        self.components = dict2AttrDict(components, shallow=True)
        add_attr(self, components)
        self.post_init()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"Attempted to get missing private attribute '{name}'")
        elif hasattr(self._ckpt, name):
            return getattr(self._ckpt, name)
        raise AttributeError(f"Attempted to get missing attribute '{name}'")

    def prev_init(self):
        """ Add some additional attributes and 
        do some pre-processing here """
        pass

    def post_init(self):
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
    def reset_model_path(self, model_path: ModelPath):
        for v in self.components.values():
            v.reset_model_path(model_path)
        self.config = set_path(self.config, model_path, max_layer=0)

    def restore(self):
        for v in self.components.values():
            v.restore()

    def save(self):
        for v in self.components.values():
            v.save()

    def _get_names(self, names: Union[str, Sequence]=None):
        if names is None:
            names = list(self.components)
        if isinstance(names, str):
            names = [names]
        assert set(names).issubset(set(self.components)), (names, self.components)
        return names
