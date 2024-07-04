from functools import partial
import os
import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import haiku as hk

from tools.pickle import save, restore
from tools.log import do_logging
from jx.elements.trainer import Trainer as TrainerBase, create_trainer
from jx.elements import optimizer
from core.typing import AttrDict, dict2AttrDict
from tools.display import print_dict_info
from tools.rms import RunningMeanStd
from tools.utils import flatten_dict, prefix_name


def construct_fake_data(env_stats, aid):
  b = 8
  s = 400
  u = len(env_stats.aid2uids[aid])
  shapes = env_stats.obs_shape[aid]
  dtypes = env_stats.obs_dtype[aid]
  action_dim = env_stats.action_dim[aid]
  basic_shape = (b, s, u)
  data = {k: jnp.zeros((b, s+1, u, *v), dtypes[k]) 
    for k, v in shapes.items()}
  data = dict2AttrDict(data)
  data.setdefault('global_state', data.obs)
  data.action = jnp.zeros((*basic_shape, action_dim), jnp.float32)
  data.value = jnp.zeros(basic_shape, jnp.float32)
  data.reward = jnp.zeros(basic_shape, jnp.float32)
  data.discount = jnp.zeros(basic_shape, jnp.float32)
  data.reset = jnp.zeros(basic_shape, jnp.float32)
  data.mu_logprob = jnp.zeros(basic_shape, jnp.float32)
  data.mu_logits = jnp.zeros((*basic_shape, action_dim), jnp.float32)
  data.advantage = jnp.zeros(basic_shape, jnp.float32)
  data.v_target = jnp.zeros(basic_shape, jnp.float32)

  print_dict_info(data)
  
  return data


class Trainer(TrainerBase):
  def add_attributes(self):
    self.popart = RunningMeanStd((0, 1), name='popart', ndim=1)
    self.indices = np.arange(self.config.n_runners * self.config.n_envs)

  def build_optimizers(self):
    theta = self.model.theta.copy()
    self.opts.theta, self.params.theta = optimizer.build_optimizer(
      params=theta, 
      **self.config.theta_opt, 
      name='theta'
    )

  def compile_train(self):
    _jit_train = jax.jit(self.theta_train, static_argnames=['debug'])
    def jit_train(*args, **kwargs):
      self.rng, rng = random.split(self.rng)
      return _jit_train(*args, rng=rng, **kwargs)
    self.jit_train = jit_train

    self.haiku_tabulate()

  def train(self, data: AttrDict):
    theta = self.model.theta.copy()
    theta, self.params.theta, stats = \
      self.jit_train(
        theta, 
        opt_state=self.params.theta, 
        data=data, 
        debug=self.config.debug
      )
    self.model.set_weights(theta)

    data = flatten_dict({k: v 
      for k, v in data.items() if v is not None}, prefix='data')
    stats = prefix_name(stats, name='train')
    stats.update(data)

    return 1, stats

  def theta_train(
    self, 
    theta, 
    rng, 
    opt_state, 
    data, 
    debug=True
  ):
    do_logging('train is traced', level='info')
    rngs = random.split(rng, 3)
    theta, opt_state, stats = optimizer.optimize(
      self.loss.loss, 
      theta, 
      opt_state, 
      kwargs={
        'rng': rngs[0], 
        'data': data, 
      }, 
      opt=self.opts.theta, 
      name='opt/theta', 
      debug=debug
    )

    return theta, opt_state, stats

  def save_optimizer(self):
    super().save_optimizer()
    self.save_popart()
  
  def restore_optimizer(self):
    super().restore_optimizer()
    self.restore_popart()

  def save(self):
    super().save()
    self.save_popart()
  
  def restore(self):
    super().restore()
    self.restore_popart()

  def get_popart_dir(self):
    path = os.path.join(self.config.root_dir, self.config.model_name)
    return path

  def save_popart(self):
    filedir = self.get_popart_dir()
    save(self.popart, filedir=filedir, filename='popart', name='popart')

  def restore_popart(self):
    filedir = self.get_popart_dir()
    self.popart = restore(
      filedir=filedir, filename='popart', 
      default=self.popart, 
      name='popart'
    )

  # def haiku_tabulate(self, data=None):
  #   rng = random.PRNGKey(0)
  #   if data is None:
  #     data = construct_fake_data(self.env_stats, 0)
  #   theta = self.model.theta.copy()
  #   print(hk.experimental.tabulate(self.theta_train)(
  #     theta, rng, self.params.theta, data
  #   ))
  #   breakpoint()


create_trainer = partial(create_trainer,
  name='il', trainer_cls=Trainer
)


if __name__ == '__main__':
  import haiku as hk
  from tools.yaml_op import load_config
  from envs.func import create_env
  from .model import create_model
  from .loss import create_loss
  from tools.log import pwc
  config = load_config('algo/ppo/configs/magw_a2c')
  config = load_config('distributed/sync/configs/smac')
  
  env = create_env(config.env)
  model = create_model(config.model, env.stats())
  loss = create_loss(config.loss, model)
  trainer = create_trainer(config.trainer, env.stats(), loss)
  data = construct_fake_data(env.stats(), 0)
  rng = random.PRNGKey(0)
  pwc(hk.experimental.tabulate(trainer.jit_train)(
    model.theta, rng, trainer.params.theta, data), color='yellow')
  # data = construct_fake_data(env.stats(), 0, True)
  # pwc(hk.experimental.tabulate(trainer.raw_meta_train)(
  #   model.eta, model.theta, trainer.params, data), color='yellow')
