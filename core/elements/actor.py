from typing import Tuple, Dict
import numpy as np
import jax

from core.elements.model import Model
from core.mixin.actor import RMS
from core.names import ANCILLARY, MODEL
from core.typing import ModelPath, AttrDict, dict2AttrDict
from tools.run import concat_along_unit_dim
from tools.utils import set_path, batch_dicts


def apply_rms_to_inp(inp, rms, update_rms):
  inp = rms.process(
    inp, 
    update_rms=update_rms, 
    mask=inp.sample_mask, 
  )
  return inp


class Actor:
  def __init__(
    self, 
    *, 
    config: AttrDict, 
    model: Model, 
    name: str
  ):
    self._raw_name = name
    self._name = f'{name}_actor'
    if config.root_dir:
      self._model_path = ModelPath(config.root_dir, config.model_name)
    else:
      self._model_path = None
    self.config = config
    self.model = model

    self.rms: RMS = self.config.rms
    self.setup_checkpoint()
    
    self._post_init()

  @property
  def name(self):
    return self._name

  def _post_init(self):
    self.gid2uids = self.model.gid2uids

  def reset_model_path(self, model_path: ModelPath):
    self._model_path = model_path
    self.config = set_path(self.config, model_path, max_layer=0)
    self.setup_checkpoint()

  def setup_checkpoint(self):
    if self.config.rms is None:
      self.config.rms = AttrDict(obs=AttrDict(), reward=AttrDict())

    self.config.rms.model_path = self._model_path
    self.config.rms.print_for_debug = self.config.get('print_for_debug', True)
    self.rms = RMS(self.config.rms, n_obs=self.model.n_groups)

  def get_raw_rms(self):
    return RMS(self.config.rms, n_obs=self.model.n_groups)

  def get_rms(self):
    return self.rms

  def __getattr__(self, name):
    # Do not expose the interface of independent elements here. 
    # Invoke them directly instead
    if name.startswith('_'):
      raise AttributeError("attempted to get missing private attribute '{}'".format(name))
    elif hasattr(self.rms, name):
      return getattr(self.rms, name)
    else:
      raise AttributeError(f"no attribute '{name}' is found")

  def __call__(self, inp: dict, evaluation: bool=False):
    """ The interface to interact with the environment
    
    Args:
      inp: input to the calling method
      evaluation: evaluation mode or not
      return_eval_stats: if return evaluation stats
    Return:
      (action, stats, rnn_state)
    """
    inp = dict2AttrDict(inp)
    inp = self._process_input(inp, evaluation)
    out = self.model.action(inp, evaluation)
    out = self._process_output(inp, out, evaluation)

    return out

  """ Overwrite the following methods if necessary """
  def _process_input(self, inp: dict, evaluation: bool):
    """ Processes input to Model at the algorithmic level 
    
    Args:
      inp: input to the model
      evaluation bool: evaluation mode or not
    Returns: 
      processed input to <model.action>
    """
    inp = self.process_obs_with_rms(inp)
    return inp

  def _process_output(
    self, 
    inp: Dict, 
    out: Tuple[Dict[str, jax.Array]], 
    evaluation: bool
  ):
    """ Post-processes output.
    
    Args:
      inp: Pre-processed inputs
      out: Model output
    Returns:
      (action, stats, rnn_state)
    """
    action, stats, state = out
    if state is not None and not evaluation:
      if isinstance(inp, (list, tuple)):
        inp = batch_dicts(inp, concat_along_unit_dim)
      stats.update({
        'state_reset': inp.state_reset, 
        'state': inp.state, 
      })
    if self.config.get('update_obs_at_execution', False) \
      and not evaluation and self.config.rms.obs.normalize_obs:
      if isinstance(inp, (list, tuple)):
        inp = batch_dicts(inp, concat_along_unit_dim)
      stats.update({k: inp[k] 
        for k in self.config.rms.obs.obs_names})
    action = jax.tree_map(np.asarray, action)
    return action, stats, state

  def process_obs_with_rms(self, obs, update_rms=None):
    if update_rms is None:
      update_rms = self.config.get('update_obs_rms_at_execution', False)
    if self.rms is not None and self.rms.is_obs_normalized:
      if isinstance(obs, (list, tuple)):
        assert len(obs) == len(self.rms.obs_rms), (len(obs), len(self.rms.obs_rms))
        obs = [apply_rms_to_inp(
          o, rms, update_rms
        ) for o, rms in zip(obs, self.rms.obs_rms)]
      else:
        obs = apply_rms_to_inp(
          obs, self.rms.obs_rms[0], update_rms=update_rms
        )
    return obs

  def normalize_obs(self, obs, is_next=False):
    if self.rms is not None:
      return self.rms.normalize_obs(obs, is_next=is_next)
    return obs

  def normalize_reward(self, reward):
    if self.rms is not None:
      return self.rms.normalize_reward(reward)
    return reward

  def update_obs_rms(self, obs, mask=None, feature_mask=None):
    obs = {k: obs[k] for k in self.rms.get_obs_names()}
    self.rms.update_obs_rms(obs, self.gid2uids, split_axis=2, 
                            mask=mask, feature_mask=feature_mask)

  def get_weights(self):
    weights = {
      MODEL: self.model.get_weights(),
      ANCILLARY: self.get_auxiliary_stats()
    }
    return weights

  def set_weights(self, weights):
    self.model.set_weights(weights[MODEL])
    if ANCILLARY in weights:
      self.set_auxiliary_stats(weights[ANCILLARY])

  def get_model_weights(self, name: str=None):
    return self.model.get_weights(name)

  def set_model_weights(self, weights):
    self.model.set_weights(weights)

  def get_auxiliary_stats(self):
    if self.rms:
      return self.rms.get_rms_stats()

  def set_auxiliary_stats(self, rms_stats):
    if self.rms:
      self.rms.set_rms_stats(rms_stats)

  def save_auxiliary_stats(self):
    """ Save the RMS and the model """
    if self.rms:
      self.rms.save_rms()

  def restore_auxiliary_stats(self):
    """ Restore the RMS and the model """
    if self.rms:
      self.rms.restore_rms()

  def save(self):
    self.model.save()
    self.save_auxiliary_stats()
  
  def restore(self):
    self.model.restore()
    self.restore_auxiliary_stats()


def create_actor(config, model, name='actor'):
  return Actor(config=config, model=model, name=name)
