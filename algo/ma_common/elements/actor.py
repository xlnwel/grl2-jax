import numpy as np

from core.elements.actor import Actor as ActorBase, apply_rms_to_inp
from core.mixin.actor import ObsRunningMeanStd, RewardRunningMeanStd
from core.typing import AttrDict, tree_slice
from tools.rms import normalize
from tools.utils import batch_dicts
from algo.ma_common.run import concat_along_unit_dim


class Actor(ActorBase):
    def setup_checkpoint(self):
        if 'obs_rms' not in self.config:
            self.config.obs_rms = AttrDict()
        if 'reward_rms' not in self.config:
            self.config.reward_rms = AttrDict()
        self.config.obs_rms.model_path = self._model_path
        self.config.reward_rms.model_path = self._model_path
        self.obs_rms = [
            ObsRunningMeanStd(self.config.obs_rms, name=f'obs_rms{i}')
            for i in range(self.model.env_stats.n_agents)
        ]
        self.reward_rms = RewardRunningMeanStd(self.config.reward_rms)

    @property
    def is_obs_or_reward_normalized(self):
        return self.obs_rms[0].is_normalized or self.reward_rms.is_normalized

    def __call__(self, inps, evaluation):
        inps = self._process_input(inps, evaluation)
        out = self.model.action(inps, evaluation)
        inp = batch_dicts(inps, concat_along_unit_dim)
        out = self._process_output(inp, out, evaluation)
        return out

    def _process_input(self, inps, evaluation):
        if self.config.obs_rms.normalize_obs:
            inps = [apply_rms_to_inp(
                inp, rms, 
                self.config.get('update_obs_rms_at_execution', False)
            ) for inp, rms in zip(inps, self.obs_rms)]
        return inps

    def get_obs_names(self):
        return self.obs_rms[0].obs_names
    
    def get_obs_rms(self, with_count=False, return_std=True):
        return [
            rms.get_rms_stats(with_count=with_count, return_std=return_std) 
            for rms in self.obs_rms
        ]

    def update_obs_rms(self, obs):
        for aid, uids in enumerate(self.model.env_stats.aid2uids):
            o = tree_slice(obs, indices=uids, axis=2)
            self.obs_rms[aid].update(o)
    
    def normalize_obs(self, obs, is_next=False):
        if self.config.obs_rms.normalize_obs:
            rms = [rms.get_rms_stats(with_count=False, return_std=True) for rms in self.obs_rms]
            rms = batch_dicts(rms, np.concatenate)
            for k, v in rms.items():
                key = f'next_{k}' if is_next else k
                val = obs[key]
                obs[key] = normalize(val, *v, clip=self.config.obs_rms.obs_clip)
        return obs

    def reset_reward_rms_return(self):
        self.reward_rms.reset_return()

    def update_reward_rms(self, reward, discount):
        self.reward_rms.update(reward, discount)

    def normalize_reward(self, reward):
        return self.reward_rms.normalize(reward)

    def process_reward_with_rms(self, reward, discount=None, update_rms=True, mask=None):
        return self.reward_rms.process(reward, discount, update_rms, mask)

    """ RMS Access & Override """
    def get_auxiliary_stats(self):
        rms = [rms.get_rms_stats() for rms in self.obs_rms]
        rms.append(self.reward_rms.get_rms_stats())
        return rms

    def set_auxiliary_stats(self, rms_stats):
        all_rms = self.obs_rms + [self.reward_rms]
        [rms.set_rms_stats(s) for rms, s in zip(all_rms, rms_stats)]

    def save_auxiliary_stats(self):
        """ Save the RMS and the model """
        for rms in self.obs_rms:
            rms.save_rms()
        self.reward_rms.save_rms()

    def restore_auxiliary_stats(self):
        """ Restore the RMS and the model """
        for rms in self.obs_rms:
            rms.restore_rms()
        self.reward_rms.restore_rms()

    def print_rms(self):
        for rms in self.obs_rms:
            rms.print_rms()
        self.reward_rms.print_rms()


def create_actor(config, model, name='actor'):
    return Actor(config=config, model=model, name=name)
