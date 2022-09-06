import haiku as hk

from nn.registry import nn_registry
from nn.utils import get_activation, get_initializer


@nn_registry.register('var')
class Variable(hk.Module):
    def __init__(
        self, 
        initializer, 
        shape=(), 
        activation=None, 
        name=None
    ):
        super().__init__(name)

        self._var = hk.get_parameter('var', shape, init=get_initializer(initializer))
        self.activation = get_activation(activation)
    
    def __call__(self):
        x = self.activation(self._var)
        return x

if __name__ == '__main__':
    def l(x):
        layer = Variable('orthogonal', shape=(3, 4))
        return layer()
    mlp = hk.transform(l)
    import jax
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (2, 3))
    params = mlp.init(rng, x)
    print(params)
    print(mlp.apply(params, None, x))

