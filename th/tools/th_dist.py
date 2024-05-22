import torch


class Categorical(torch.distributions.Categorical):
  def stop_gradient(self):
    logits = None if self.logits is None else self.logits.detach()
    probs = None if self.probs is None else self.probs.detach()
    super().__init__(logits=logits, probs=probs)

  def get_stats(self, prefix=None):
    if prefix is None:
      return {'logits': self.logits}
    else:
      return {f'{prefix}_logits': self.logits}
  
  @staticmethod
  def stats_keys(prefix=None):
    if prefix is None:
      return ('logits',)
    else:
      return (f'{prefix}_logits',)


class MultivariateNormalDiag(torch.distributions.Normal):
  def __init__(self, loc, scale=None, joint_log_prob=True):
    super().__init__(loc, scale)
    self._joint_log_prob = joint_log_prob

  def stop_gradient(self):
    loc = self.loc.detach()
    scale = self.scale.detach()
    super().__init__(loc, scale)

  def log_prob(self, value, joint=None):
    if joint is None:
      joint = self._joint_log_prob
    logp = super().log_prob(value)
    if joint:
      return logp.sum(-1)
    else:
      return logp

  def entropy(self):
    return super().entropy().sum(-1)

  def mode(self):
    return self.mean

  def get_stats(self, prefix=None):
    if prefix is None:
      stats= {
        'loc': self.mean, 
        'scale': self.stddev, 
      }
      stats.update({
        f'loc{i}': self.mean[..., i] for i in range(self.mean.shape[-1])
      })
      stats.update({
        f'scale{i}': self.stddev[..., i] for i in range(self.stddev.shape[-1])
      })
    else:
      stats= {
        f'{prefix}_loc': self.mean, 
        f'{prefix}_scale': self.stddev, 
      }
      stats.update({
        f'{prefix}_loc{ i}': self.mean[..., i] for i in range(self.mean.shape[-1])
      })
      stats.update({
        f'{prefix}_scale{i}': self.stddev[..., i] for i in range(self.stddev.shape[-1])
      })
    return stats

  @staticmethod
  def stats_keys(prefix=None):
    if prefix is None:
      return ('loc', 'scale')
    else:
      return (f'{prefix}mean', f'{prefix}_scale')
