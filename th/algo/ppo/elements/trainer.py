from functools import partial
import numpy as np
from torch.utils._pytree import tree_map

from th.core.names import TRAIN_AXIS
from th.core.elements.trainer import TrainerBase, create_trainer
from th.core import optimizer
from th.core.typing import AttrDict, dict2AttrDict
from th.tools.th_utils import to_tensor, to_numpy
from tools.display import print_dict_info
from tools.utils import flatten_dict, prefix_name


def construct_fake_data(env_stats, aid):
  b = 8
  s = 400
  u = len(env_stats.aid2uids[aid])
  shapes = env_stats.obs_shape[aid]
  dtypes = env_stats.obs_dtype[aid]
  action_dim = env_stats.action_dim[aid]
  basic_shape = (b, s, u)
  data = {k: np.zeros((b, s+1, u, *v), dtypes[k]) 
    for k, v in shapes.items()}
  data = dict2AttrDict(data)
  data.setdefault('global_state', data.obs)
  data.action = np.zeros((*basic_shape, action_dim), np.float32)
  data.value = np.zeros(basic_shape, np.float32)
  data.reward = np.zeros(basic_shape, np.float32)
  data.discount = np.zeros(basic_shape, np.float32)
  data.reset = np.zeros(basic_shape, np.float32)
  data.mu_logprob = np.zeros(basic_shape, np.float32)
  data.mu_logits = np.zeros((*basic_shape, action_dim), np.float32)
  data.advantage = np.zeros(basic_shape, np.float32)
  data.v_target = np.zeros(basic_shape, np.float32)

  print_dict_info(data)
  data = to_tensor(data)
  
  return data


class Trainer(TrainerBase):
  def add_attributes(self):
    self.indices = np.arange(self.config.n_runners * self.config.n_envs)

  def build_optimizers(self):
    theta = self.model.theta
    self.clip_norm = self.config.pop('clip_norm')
    if self.config.get('theta_opt'):
      self.opts.theta = optimizer.build_optimizer(
        params=theta, 
        **self.config.theta_opt, 
      )
    else:
      self.opts.policy = optimizer.build_optimizer(
        params=theta.policy, 
        **self.config.policy_opt, 
      )
      self.opts.value = optimizer.build_optimizer(
        params=theta.value, 
        **self.config.value_opt, 
      )

  def train(self, data: AttrDict):
    if self.config.n_runners * self.config.n_envs < self.config.n_mbs:
      self.indices = np.arange(self.config.n_mbs)
      data = tree_map(
        lambda x: x.reshape(self.config.n_mbs, -1, *x.shape[2:]), data)

    self.model.train()
    all_stats = AttrDict()
    v_target = None
    for e in range(self.config.n_epochs):
      stats = self.theta_train(data=data)
      v_target = stats.raw_v_target
      # print_dict_info(stats)
      if e == self.config.n_epochs-1:
        all_stats.update(**prefix_name(stats, name=f'group_last_epoch'))
      elif e == 0:
        all_stats.update(**prefix_name(stats, name=f'group_first_epoch'))

    if self.config.popart:
      self.model.vnorm.update(to_numpy(v_target))
    all_stats = to_numpy(all_stats)
    data = flatten_dict(
      {k: v for k, v in data.items() if v is not None}, prefix='data')
    all_stats.update(data)

    return all_stats

  def theta_train(self, data):
    data = to_tensor(data, self.tpdv)
    if self.config.get('theta_opt'):
      actor_loss, value_loss, stats = self.loss.loss(data)
      loss = actor_loss + value_loss
      norm = optimizer.optimize(
        self.opts.theta, loss, self.model.theta, self.clip_norm
      )
      stats.norm = norm
    else:
      policy_loss, value_loss, stats = self.loss.loss(data)
      policy_norm = optimizer.optimize(
        self.opts.policy, policy_loss, self.model.theta.policy, self.clip_norm
      )
      value_norm = optimizer.optimize(
        self.opts.value, value_loss, self.model.theta.value, self.clip_norm
      )
      stats.policy_norm = policy_norm
      stats.value_norm = value_norm
      
    return stats


create_trainer = partial(create_trainer,
  name='ppo', trainer_cls=Trainer
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
