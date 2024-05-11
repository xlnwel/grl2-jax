import jax
import haiku as hk


def init_module(ModuleCls, config, *args, **kwargs):
  def build(*args, **kwargs):
    module = ModuleCls(**config)
    return module(*args, **kwargs)
  net = hk.transform(build)
  rng = jax.random.PRNGKey(42)
  params = net.init(rng, *args, **kwargs)
  return params, net


def viz_module(ModuleCls, config, *args, **kwargs):
  params, net = init_module(ModuleCls, config, *args, **kwargs)
  rng = jax.random.PRNGKey(42)
  dot = hk.experimental.to_dot(net.apply)(params, rng, *args, **kwargs)
  import graphviz
  graphviz.Source(dot, filename='dot.gv', directory='.')


if __name__ == '__main__':
  from jax import numpy as jnp
  from algo.ppo.elements.nn import Policy
  from tools.yaml_op import load_config
  print(load_config('algo/ppo/configs/grf'))
  config = load_config('algo/ppo/configs/grf').model.policy
  config.pop('nn_id')
  config.is_action_discrete = {'action': True}
  config.action_dim = {'action': 10}
  x = jnp.zeros((10, 20))
  viz_module(Policy, config, x)
