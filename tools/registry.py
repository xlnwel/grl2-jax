import functools

from tools.dummy import Dummy


class Registry:
    def __init__(self, name, DummyFunc=Dummy):
        self.name = name
        self._mapping = {None: DummyFunc}

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
