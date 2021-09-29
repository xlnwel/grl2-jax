import numpy as np

from core.mixin.actor import RMS
from core.module import Actor
from utility.utils import concat_map


class PPOActor(Actor):
    def _post_init(self, config):
        config['rms']['root_dir'] = config['root_dir']
        config['rms']['model_name'] = config['model_name']
        self.rms = RMS(config['rms'])

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.rms, name)

    def _process_input(self, inp: dict, evaluation: bool):
        def split_input(inp):
            actor_state, value_state = self.model.split_state(inp['state'])
            actor_inp = dict(
                obs=inp['obs'],
                state=actor_state,
                mask=inp['mask'],
                action_mask=inp['action_mask']
            )
            value_inp = dict(
                global_state=inp['global_state'],
                state=value_state,
                mask=inp['mask']
            )
            return {'actor_inp': actor_inp, 'value_inp': value_inp}
        if evaluation:
            inp = self.rms.process_obs_with_rms(inp, update_rms=False)
        else:
            life_mask = inp.get('life_mask')
            inp = self.rms.process_obs_with_rms(inp, mask=life_mask)
        inp, tf_inp = super()._process_input(inp, evaluation)
        tf_inp = split_input(tf_inp)
        return inp, tf_inp

    def _process_output(self, inp, out, evaluation):
        out = super()._process_output(inp, out, evaluation)

        if not evaluation:
            out[1].update({
                'obs': inp['obs'],
                'global_state': inp['global_state'],
                'mask': inp['mask'], 
            })
            if 'action_mask' in inp:
                out[1]['action_mask'] = inp['action_mask']
            if 'life_mask' in inp:
                out[1]['life_mask'] = inp['life_mask']

        return out

    def get_rms_stats(self):
        obs_rms, rew_rms = self.rms.get_rms_stats()
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

    def save_auxiliary_stats(self):
        """ Save the RMS and the model """
        self.rms.save_rms()

    def restore_auxiliary_stats(self):
        """ Restore the RMS and the model """
        self.rms.restore_rms()

def create_actor(config, model, name='ppo'):
    return PPOActor(config=config, model=model, name=name)
