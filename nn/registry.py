import functools

from nn.utils import Dummy


class Registry:
    def __init__(self, name):
        self.name = name
        self._mapping = {None: Dummy}
    
    def register(self, name: str):
        def _thunk(func):
            self._mapping[name] = func
            return func
        return _thunk
    
    def get(self, name: str):
        if name not in self._mapping:
            raise ValueError(f'{name} is not registered in {self.name} registry')
        return self._mapping[name]

    def contain(self, name: str):
        return name in self._mapping
    
    def get_all(self):
        return self._mapping


def register_all(registry, globs):
    for k, v in globs.items():
        if isinstance(v, functools.partial):
            registry.register(k)(v)


layer_registry = Registry(name='layer')
am_registry = Registry(name='am') # convolutional attention modules
block_registry = Registry(name='block')
subsample_registry = Registry(name='subsample')
cnn_registry = Registry(name='cnn')
rnn_registry = Registry(name='rnn')
nn_registry = Registry(name='nn')
nn_registry.register('cnn')(cnn_registry)
nn_registry.register('rnn')(rnn_registry)
