import jax.numpy as jnp
import haiku as hk

from nn.registry import layer_registry
from nn.utils import get_initializer, get_activation, \
    call_norm, calculate_scale, FixedInitializer


@layer_registry.register('layer')
class Layer:
    def __init__(
        self, 
        *args, 
        layer_type='linear', 
        norm=None, 
        activation=None, 
        w_init='glorot_uniform', 
        b_init='zeros', 
        name=None, 
        norm_after_activation=False, 
        norm_kwargs={
            'axis': -1, 
            'create_scale': True, 
            'create_offset': True, 
        }, 
        **kwargs
    ):
        self.name = name or layer_type
        self.layer_cls = layer_registry.get(layer_type)
        self.layer_args = args
        scale = kwargs.pop('scale', calculate_scale(activation))
        self.w_init = get_initializer(w_init, scale=scale)
        self.b_init = get_initializer(b_init)
        self.layer_kwargs = kwargs

        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self._norm_after_activation = norm_after_activation
        self.activation = get_activation(activation)

    def __call__(self, x, is_training=True, **kwargs):
        if self.layer_args:
            x = self.layer_cls(
                *self.layer_args, 
                w_init=self.w_init, 
                b_init=self.b_init, 
                name=self.name, 
                **self.layer_kwargs
            )(x)
        
        if not self._norm_after_activation:
            x = call_norm(self.norm, self.norm_kwargs, x, is_training=is_training)
        if self.activation is not None:
            x = self.activation(x)
        if self._norm_after_activation:
            x = call_norm(self.norm, self.norm_kwargs, x, is_training=is_training)

        return x


@layer_registry.register('elayer')
class ELayer(hk.Module):
    def __init__(
        self, 
        out_size, 
        ensemble_size, 
        with_bias=True, 
        w_init='glorot_uniform', 
        b_init='zeros', 
        scale=1, 
        expand_edim=False, 
        name=None
    ):
        super().__init__(name)

        self.ensemble_size = ensemble_size
        self.out_size = out_size
    
        self.with_bias = with_bias
        self.w_init = get_initializer(w_init, scale=scale)
        self.b_init = get_initializer(b_init)
        self.expand_edim = expand_edim

    def __call__(self, x):
        w, b = self.build_net(x)
        if self.expand_edim:
            # The last two dims are used for standard matrix oprations. 
            # The antepenultimate is the ensemble dimension, same for w and b
            x = jnp.expand_dims(x, -3)
        assert (x.shape[-3] == 1) or (x.shape[-3] == self.ensemble_size), (x.shape, self.ensemble_size)
        x = x @ w + b
        return x
    
    @hk.transparent
    def build_net(self, x):
        in_size = x.shape[-1]
        inits = [self.w_init((in_size, self.out_size), x.dtype) 
            for _ in range(self.ensemble_size)]
        init = jnp.stack(inits)
        init_shape = init.shape
        init = FixedInitializer(init)
        w = hk.get_parameter('w', shape=init_shape, init=init)
        if self.with_bias:
            init_shape = (self.ensemble_size, 1, self.out_size)
            init = get_initializer('zeros')
            b = hk.get_parameter('b', shape=init_shape, init=init)
        else:
            b = 0

        return w, b


layer_registry.register('linear')(hk.Linear)

if __name__ == '__main__':
    import jax
    def f(x):
        def l(x):
            layer = Layer(3, w_init='orthogonal', scale=.01, 
                activation='relu', norm='layer', name='layer')
            return layer(x)
        mlp = hk.transform(l)
        rng = jax.random.PRNGKey(42)
        params = mlp.init(rng, x)
        return params, mlp
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (2, 3)) 
    params, mlp = f(x)
    print(params)
    print(mlp.apply(params, None, x))
    print(hk.experimental.tabulate(mlp)(x))
    def g(params, x):
        rng = jax.random.PRNGKey(42)
        y = mlp.apply(params, rng, x)
        return y
    import graphviz
    dot = hk.experimental.to_dot(g)(params, x)
    graphviz.Source(dot)