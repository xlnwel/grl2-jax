import numpy as np

from utility.tf_utils import tensor2numpy, numpy2tensor
from algo.ppo.elements.actor import PPOActor


class MAPPOActor(PPOActor):
    """ Calling Methods """
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
        tf_inp = numpy2tensor(inp)
        tf_inp = split_input(tf_inp)
        return inp, tf_inp

    def _process_output(self, inp, out, evaluation):
        action, terms, state = out
        action, terms, prev_state = tensor2numpy((action, terms, inp['state']))

        if not evaluation:
            terms.update({
                'obs': inp['obs'],
                'global_state': inp['global_state'],
                'mask': inp['mask'], 
                **prev_state._asdict(),
            })
            if 'action_mask' in inp:
                terms['action_mask'] = inp['action_mask']
            if 'life_mask' in inp:
                terms['life_mask'] = inp['life_mask']

        return action, terms, state


def create_actor(config, model, name='mappo'):
    return MAPPOActor(config=config, model=model, name=name)
