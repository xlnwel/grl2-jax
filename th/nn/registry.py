import functools

from tools.registry import Registry

class Dummy:
  def get_weights(self):
    pass

  def set_weights(self, weights):
    pass


Registry = functools.partial(Registry, DummyFunc=Dummy)

layer_registry = Registry(name='layer')
# am_registry = Registry(name='am') # convolutional attention modules
# block_registry = Registry(name='block')
# subsample_registry = Registry(name='subsample')
# cnn_registry = Registry(name='cnn')
# rnn_registry = Registry(name='rnn')
nn_registry = Registry(name='nn')
# nn_registry.register('cnn')(cnn_registry)
# nn_registry.register('rnn')(rnn_registry)
