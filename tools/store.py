from core.log import do_logging


class StateStore:
  state = {}

  def __init__(self, name, constructor, enter_set_fn, exit_set_fn=None):
    """ Temporary state store for get and restore the state of memory, environment and etc.
    Params:
      constructor (func): a function that constructs the initial state
      enter_set_fn (func): a function that sets the state and returns the current statea
    """
    self._name = name
    self._constructor = constructor
    self._enter_set_fn = enter_set_fn
    self._exit_set_fn = enter_set_fn if exit_set_fn is None else exit_set_fn
    self._tmp_state = None

  @property
  def name(self):
    return self._name

  def __enter__(self):
    if self._name not in self.state:
      do_logging(f'Building state for "{self._name}"...')
      self.state[self._name] = self._constructor()
    self._tmp_state = self._enter_set_fn(self.state[self._name])
    return self
  
  def __exit__(self, exc_type, exc_value, traceback):
    """ store state at exit """
    self.state[self._name] = self._exit_set_fn(self._tmp_state)


class TempStore:
  def __init__(self, constructor, enter_set_fn, exit_set_fn=None):
    self._constructor = constructor
    self._enter_set_fn = enter_set_fn
    self._exit_set_fn = enter_set_fn if exit_set_fn is None else exit_set_fn

  def __enter__(self):
    self.state = self._enter_set_fn(self._constructor())
    return self
  
  def __exit__(self, exc_type, exc_value, traceback):
    self.state = self._exit_set_fn(self.state)
