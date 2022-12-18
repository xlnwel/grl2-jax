from core.log import do_logging


class StateStore:
    state = {}

    def __init__(self, name, constructor, get_fn, set_fn):
        """ Temporary stete store for get and restore the state of memory, environment and etc.
        Params:
            constructor (func): a function that constructs the initial state
            get_fn (func): a function that retrieves the state
            set_fn (func): a function that set the state
        """
        self._name = name
        self._constructor = constructor
        self._get_fn = get_fn
        self._set_fn = set_fn

    @property
    def name(self):
        return self._name

    def __enter__(self):
        if self._name not in self.state:
            do_logging(f'Building state for "{self._name}"...')
            self.state[self._name] = self._constructor()
        self._set_fn(self.state[self._name])
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """ store state at exit """
        self.state[self._name] = self._get_fn()


class TempStore:
    def __init__(self, get_fn, set_fn):
        self._get_fn = get_fn
        self._set_fn = set_fn

    def __enter__(self):
        self.state = self._get_fn()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._set_fn(self.state)
