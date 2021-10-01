import numpy as np

from core.module import Actor
from algo.ppo.elements.actor import PPOActor


class MAPPOActor(PPOActor):
    def _process_input(self, inp: dict, evaluation: bool):
        def concat_map_except_state(inp):
            for k, v in inp.items():
                if k != 'state':
                    inp[k] = np.concatenate(v)
            return inp

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
        
        inp = concat_map_except_state(inp)
        if evaluation:
            inp = self.rms.process_obs_with_rms(inp, update_rms=False)
        else:
            life_mask = inp.get('life_mask')
            inp = self.rms.process_obs_with_rms(inp, mask=life_mask)
        inp, tf_inp = Actor._process_input(self, inp, evaluation)
        tf_inp = split_input(tf_inp)
        return inp, tf_inp

    def _process_output(self, inp, out, evaluation):
        out = Actor._process_output(self, inp, out, evaluation)

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


def create_actor(config, model, name='mappo'):
    return MAPPOActor(config=config, model=model, name=name)
