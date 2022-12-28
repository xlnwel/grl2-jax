import haiku as hk

from nn.registry import nn_registry
from nn.utils import get_initializer


@nn_registry.register('embed')
class Embedding(hk.Module):
    def __init__(self, name='embed', **config):
        super().__init__(name=name)
        config = config.copy()
        self.vocab_size = config.pop('vocab_size')
        self.embed_dim = config.pop('embed_dim')
        w_init = config.pop('w_init', 'truncated_normal')
        self.w_init = get_initializer(w_init)

    def __call__(
        self, 
        x, 
        multiply: bool=False, 
    ):
        """
        Args:
            tile: If true we replicate the input's last dimension, 
                this yields a tensor of shape (B, U, U, D) given 
                the input/resulting embedding of shape (B, U)/(B, U, D).
                This is useful in MARL, where we consider other agents'
                actions for the current agent.
            mask_out_self: If true (and <tile> must be true), we 
                make (B, i, i, D) = 0 for i in range(A)
            flatten: Flatten the tiled tensor.
        """
        embed = hk.get_parameter('embed', (self.vocab_size, self.embed_dim), init=self.w_init)
        if multiply:
            x = x @ embed
        else:
            x = embed[x]
        return x


if __name__ == '__main__':
    import jax
    def layer(x, multiply):
        layer = Embedding(vocab_size=10, embed_dim=1)
        return layer(x, multiply)
    rng = jax.random.PRNGKey(42)
    x = jax.random.randint(rng, (2, 3), 0, 10)
    net = hk.transform(layer)
    params = net.init(rng, x, False)
    print(params)
    print(net.apply(params, None, x, False))
    # print(hk.experimental.tabulate(net)(x, False))
    x = jax.random.uniform(rng, (2, 10))
    net = hk.transform(layer)
    params = net.init(rng, x, True)
    print(params)
    print(net.apply(params, None, x, True))
    # print(hk.experimental.tabulate(net)(x, True))