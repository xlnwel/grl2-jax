from functools import partial
import os
import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import haiku as hk

from core.ckpt.pickle import save, restore
from core.log import do_logging
from core.names import TRAIN_AXIS
from core.elements.trainer import TrainerBase, create_trainer
from core import optimizer
from core.typing import AttrDict
from tools.rms import RunningMeanStd
from tools.utils import flatten_dict, prefix_name, yield_from_tree_with_indices
from algo.ppo.elements.trainer import construct_fake_data


class Trainer(TrainerBase):
  def add_attributes(self):
    self.gids = self.model.gids
    self.popart = [
      RunningMeanStd((0, 1), name=f'popart{i}', ndim=1) 
      for i, _ in enumerate(self.gids)
    ]
    self.indices = np.arange(self.config.n_runners * self.config.n_envs)
    self.gid2uids = self.model.gid2uids

    self.n_epochs = self.config.n_epochs
    self.n_mbs = self.config.n_mbs

  def build_optimizers(self):
    theta = self.model.theta.copy()
    if self.config.get('theta_opt'):
      self.opts.theta, self.params.theta = [list(x)
        for x in zip(*[optimizer.build_optimizer(
        params=AttrDict(policy=p, value=v), 
        **self.config.theta_opt, 
        name=f'theta{i}'
      ) for i, (p, v) in enumerate(zip(theta.policies, theta.vs))])]
    else:
      self.params.theta = AttrDict()
      self.opts.policies, self.params.theta.policies = [list(x)
        for x in zip(*[optimizer.build_optimizer(
        params=p, 
        **self.config.policy_opt, 
        name=f'policy{i}'
      ) for i, p in enumerate(theta.policies)])]
      self.opts.vs, self.params.theta.vs = [list(x)
        for x in zip(*[optimizer.build_optimizer(
        params=v, 
        **self.config.value_opt, 
        name=f'value{i}'
      ) for i, v in enumerate(theta.vs)])]

  def compile_train(self):
    _jit_train = jax.jit(self.theta_train, 
      static_argnames=['gid', 'compute_teammate_log_ratio', 'debug'])
    def jit_train(*args, **kwargs):
      self.rng, rng = random.split(self.rng)
      return _jit_train(*args, rng=rng, **kwargs)
    self.jit_train = jit_train

    self.haiku_tabulate()

  def train(self, data: AttrDict, gids=None):
    if self.config.n_runners * self.config.n_envs < self.config.n_mbs:
      self.indices = np.arange(self.config.n_mbs)
      data = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (self.config.n_mbs, -1, *x.shape[TRAIN_AXIS.UNIT:])), data)

    theta = self.model.theta.copy()
    opt_state = self.params.theta

    if self.config.update_scheme == 'step':
      theta, opt_state, stats = \
        self.stepwise_sequential_opt(
          theta, opt_state, data, self.n_epochs, 
          self.n_mbs, self.indices, 
          self.jit_train, gids=gids, 
          return_stats=True, name=None
        )
    elif self.config.update_scheme == 'whole':
      theta, opt_state, stats = \
        self.sequential_opt(
          theta, opt_state, data, self.n_epochs, 
          self.n_mbs, self.indices, 
          self.jit_train, gids=gids, 
          return_stats=True, name=None
        )
    else:
      raise NotImplementedError(self.config.update_scheme)

    self.model.set_weights(theta)
    self.params.theta = opt_state

    if self.config.popart:
      for gid, uids in enumerate(self.gid2uids):
        self.popart[gid].update(np.take(stats.v_target, indices=uids, axis=TRAIN_AXIS.UNIT))
      stats['theta/popart/mean'] = [rms.mean for rms in self.popart]
      stats['theta/popart/std'] = [rms.std for rms in self.popart]

    data = flatten_dict({k: v 
      for k, v in data.items() if v is not None}, prefix='data')
    stats.update(data)

    return self.n_epochs * self.n_mbs, stats

  def sequential_opt(self, theta, opt_state, data, 
      n_epochs, n_mbs, indices, train_fn, gids=None, 
      return_stats=True, name=None):
    teammate_log_ratio = jnp.zeros((*data.mu_logprob.shape[:TRAIN_AXIS.UNIT], 1))

    v_target = [None for _ in self.gids]
    all_stats = AttrDict()
    if gids is None:
      gids = np.random.permutation(self.gids)
    for gid in gids:
      gid = int(gid)
      uids = self.gid2uids[gid]
      agent_theta, agent_opt_state = get_params_and_opt(theta, opt_state, gid)
      agent_data = data.slice(indices=uids, axis=TRAIN_AXIS.UNIT)
      for e in range(n_epochs):
        vts = []
        np.random.shuffle(indices)
        _indices = np.split(indices, n_mbs)
        for i, d in enumerate(yield_from_tree_with_indices(
            agent_data, _indices, axis=0)):
          if self.config.popart:
            d.popart_mean = self.popart[gid].mean
            d.popart_std = self.popart[gid].std
          tlr = teammate_log_ratio[_indices[i]]
          agent_theta, agent_opt_state, stats = \
            train_fn(
              agent_theta, 
              opt_state=agent_opt_state, 
              data=d, 
              teammate_log_ratio=tlr, 
              gid=gid, 
              compute_teammate_log_ratio=False, 
              debug=self.config.get('debug', False)
            )
          vts.append(stats.pop('raw_v_target'))
          if e == 0 and i == 0:
            all_stats.update(**prefix_name(stats, name=f'group{gid}_first_epoch'))
        if e == n_epochs-1:
          all_stats.update(**prefix_name(stats, name=f'group{gid}_last_epoch'))
      teammate_log_ratio = self.compute_teammate_log_ratio(
        agent_theta.policy, self.rng, teammate_log_ratio, agent_data
      )
      
      v_target[gid] = np.concatenate(vts)
      all_stats[f'group{gid}/teammate_log_ratio'] = teammate_log_ratio
      theta, opt_state = set_params_and_opt(
        theta, opt_state, gid, agent_theta, agent_opt_state)
    
    if return_stats:
      all_stats.v_target = np.concatenate(v_target, axis=TRAIN_AXIS.UNIT)
      assert all_stats.v_target.shape == data.reward.shape, (all_stats.v_target.shape, data.reward.shape)
      all_stats = prefix_name(all_stats, name=name)
      return theta, opt_state, all_stats
    return theta, opt_state

  def stepwise_sequential_opt(self, theta, opt_state, data, 
      n_epochs, n_mbs, indices, train_fn, gids=None, 
      return_stats=True, name=None):
    all_stats = AttrDict()

    for e in range(n_epochs):
      # indices = random.shuffle(shuffle_rng[e], indices)
      np.random.shuffle(indices)
      _indices = np.split(indices, n_mbs)
      v_target = []
      for i, data_slice in enumerate(
          yield_from_tree_with_indices(data, _indices, axis=TRAIN_AXIS.BATCH)):
        vts = [None for _ in self.gid2uids]
        teammate_log_ratio = jnp.zeros((*data_slice.mu_logprob.shape[:TRAIN_AXIS.UNIT], 1))
        if gids is None:
          gids = np.random.permutation(self.gids)
        uids = [self.gid2uids[gid] for gid in gids]
        for gid, agent_data in zip(gids, 
            yield_from_tree_with_indices(data_slice, uids, axis=TRAIN_AXIS.UNIT)):
          agent_theta, agent_opt_state = get_params_and_opt(theta, opt_state, gid)
          if self.config.popart:
            agent_data.popart_mean = self.popart[gid].mean
            agent_data.popart_std = self.popart[gid].std
          agent_theta, agent_opt_state, stats = \
            train_fn(
              agent_theta, 
              opt_state=agent_opt_state, 
              data=agent_data, 
              teammate_log_ratio=teammate_log_ratio, 
              gid=gid,
              compute_teammate_log_ratio=True, 
              debug=self.config.get('debug', False)
            )
          teammate_log_ratio = stats.teammate_log_ratio

          theta, opt_state = set_params_and_opt(
            theta, opt_state, gid, agent_theta, agent_opt_state)
          
          if return_stats:
            vts[gid] = stats.pop('raw_v_target')
            if e == 0 and i == 0:
              all_stats.update(**prefix_name(stats, name=f'agent{gid}_first_epoch'))
            elif e == n_epochs-1 and i == n_mbs - 1:
              all_stats.update(**prefix_name(stats, name=f'agent{gid}_last_epoch'))
        v_target.append(vts)

    if return_stats:
      v_target = [np.concatenate(v, TRAIN_AXIS.UNIT) for v in v_target]
      all_stats.v_target = np.concatenate(v_target)
      assert all_stats.v_target.shape == data.reward.shape, (all_stats.v_target.shape, data.reward.shape)
      all_stats = prefix_name(all_stats, name=name)
      return theta, opt_state, all_stats
    return theta, opt_state

  def theta_train(
    self, 
    theta, 
    rng, 
    opt_state, 
    data, 
    teammate_log_ratio, 
    gid, 
    compute_teammate_log_ratio=True, 
    debug=True
  ):
    do_logging('train is traced', backtrack=4)
    rngs = random.split(rng, 3)
    if self.config.get('theta_opt'):
      theta, opt_state, stats = optimizer.optimize(
        self.loss.loss, 
        theta, 
        opt_state, 
        kwargs={
          'rng': rngs[0], 
          'data': data, 
          'teammate_log_ratio': teammate_log_ratio,
        }, 
        opt=self.opts.theta[gid], 
        name='opt/theta', 
        debug=debug
      )
    else:
      theta.value, opt_state.value, stats = optimizer.optimize(
        self.loss.value_loss, 
        theta.value, 
        opt_state.value, 
        kwargs={
          'rng': rngs[0], 
          'policy_theta': theta.policy, 
          'data': data, 
          'teammate_log_ratio': teammate_log_ratio
        }, 
        opt=self.opts.vs[gid], 
        name='opt/value', 
        debug=debug
      )
      theta.policy, opt_state.policy, stats = optimizer.optimize(
        self.loss.policy_loss, 
        theta.policy, 
        opt_state.policy, 
        kwargs={
          'rng': rngs[1], 
          'data': data, 
          'stats': stats,
        }, 
        opt=self.opts.policies[gid], 
        name='opt/policy', 
        debug=debug
      )

    if compute_teammate_log_ratio:
      stats.teammate_log_ratio = self.compute_teammate_log_ratio(
        theta.policy, rngs[2], teammate_log_ratio, data)

    inverse_mu = 1 / jnp.exp(data.mu_logprob)
    if not debug:
      stats = AttrDict(
        ratio=stats.ratio, 
        log_ratio=stats.log_ratio, 
        reg=stats.reg, 
        reg_loss=stats.reg_loss, 
        pos_sample_reg_loss=stats.pos_sample_reg_loss, 
        sample_reg_loss=stats.sample_reg_loss, 
        raw_sample_reg_grads=stats.raw_sample_reg_grads, 
        reg_below_threshold=stats.reg_below_threshold, 
        reg_above_threshold=stats.reg_above_threshold, 
        inverse_mu=inverse_mu, 
        clip_frac=stats.clip_frac, 
        entropy=stats.entropy, 
        raw_v_target=stats.raw_v_target, 
        v_target=stats.v_target, 
        teammate_log_ratio=stats.teammate_log_ratio, 
        adv_ratio_pp=stats.adv_ratio_pp, 
        adv_ratio_pn=stats.adv_ratio_pn, 
        adv_ratio_np=stats.adv_ratio_np, 
        adv_ratio_nn=stats.adv_ratio_nn, 
        pn_ratio=stats.pn_ratio, 
        np_ratio=stats.np_ratio, 
      )
    else:
      stats.inverse_mu = inverse_mu
    return theta, opt_state, stats

  def compute_teammate_log_ratio(
      self, policy_params, rng, teammate_log_ratio, data):
    if 'popart_mean' in data:
      data.pop('popart_mean')
      data.pop('popart_std')
    pi_logprob = self.model.action_logprob(policy_params, rng, data)
    log_ratio = pi_logprob - data.mu_logprob
    if log_ratio.ndim > 3:
      # for continuous actions, we sum up the log ratio along the action dimension 
      log_ratio = jnp.sum(log_ratio, axis=3)
    if log_ratio.shape[TRAIN_AXIS.UNIT] > 1:
      # for groups contains several units, the log ratios are summed up along the unit dimension
      log_ratio = jnp.sum(log_ratio, axis=TRAIN_AXIS.UNIT, keepdims=True)
    teammate_log_ratio += log_ratio
  
    return teammate_log_ratio
  
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

def get_params_and_opt(params, opt_state, gid):
  agent_params = AttrDict(
    policy=params.policies[gid], value=params.vs[gid])
  if isinstance(opt_state, list):
    agent_opt_state = opt_state[gid]
  else:
    agent_opt_state = AttrDict(
      policy=opt_state.policies[gid], value=opt_state.vs[gid])
  
  return agent_params, agent_opt_state

def set_params_and_opt(params, opt_state, gid, agent_params, agent_opt_state):
  params.policies[gid] = agent_params.policy
  params.vs[gid] = agent_params.value
  if isinstance(opt_state, list):
    opt_state[gid] = agent_opt_state
  else:
    opt_state.policies[gid] = agent_opt_state.policy
    opt_state.vs[gid] = agent_opt_state.value

  return params, opt_state


create_trainer = partial(create_trainer,
  name='happo', trainer_cls=Trainer
)


if __name__ == '__main__':
  import haiku as hk
  from tools.yaml_op import load_config
  from env.func import create_env
  from .model import create_model
  from .loss import create_loss
  from core.log import pwc
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
