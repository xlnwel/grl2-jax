import os
import tensorflow as tf

from core.module import Model
from core.mixin import RMS, Actor
from utility.file import source_file

source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class PPOModel(Model, RMS, Actor):
    def _post_init(self, config):
        self._setup_rms_stats()

    def _process_input(self, env_output, evaluation):
        obs = self._process_obs(env_output.obs, update_rms=not evaluation)
        return obs, {}

    def _process_output(self, obs, kwargs, out, evaluation):
        out = super()._process_output(obs, kwargs, out, evaluation)
        if self._normalize_obs and not evaluation:
            terms = out[1]
            terms['obs'] = obs
        return out

    @tf.function
    def action(self, obs, evaluation=False, return_value=True, **kwargs):
        x = self.encode(obs)
        act_dist = self.actor(x, evaluation=evaluation)
        action = self.actor.action(act_dist, evaluation)

        if evaluation:
            return action
        else:
            logpi = act_dist.log_prob(action)
            if return_value:
                if hasattr(self, 'value_encoder'):
                    x = self.value_encoder(obs)
                value = self.value(x)
                terms = {'logpi': logpi, 'value': value}
            else:
                terms = {'logpi': logpi}
            return action, terms    # keep the batch dimension for later use

    @tf.function
    def compute_value(self, x):
        if hasattr(self, 'value_encoder'):
            x = self.value_encoder(x)
        else:
            x = self.encoder(x)
        value = self.value(x)
        return value

    def encode(self, x):
        return self.encoder(x)

    def get_rms_stats(self):
        obs_rms, rew_rms = super().get_rms_stats()
        stats = {}
        if rew_rms:
            stats.update({
                'train/reward_rms_mean': rew_rms.mean,
                'train/reward_rms_var': rew_rms.var
            })
        if obs_rms:
            for k, v in obs_rms.items():
                stats.update({
                    f'train/{k}_rms_mean': v.mean,
                    f'train/{k}_rms_var': v.var,
                })
        return stats

    def save(self, print_terminal_info=False):
        """ Save the RMS and the model """
        self.save_rms()
        super().save(print_terminal_info=print_terminal_info)

    def restore(self):
        """ Restore the RMS and the model """
        self.restore_rms()
        super().restore()


def create_model(config, env_stats, name='ppo'):
    config['actor']['action_dim'] = env_stats.action_dim
    config['actor']['is_action_discrete'] = env_stats.action_dim

    return PPOModel(config=config, name=name)
