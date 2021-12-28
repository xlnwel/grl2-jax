import tensorflow as tf

from core.elements.actor import Actor
from utility.tf_utils import numpy2tensor, tensor2numpy


class MAPPOActor(Actor):
    """ Calling Methods """
    def _process_input(self, inp: dict, evaluation: bool):
        def split_input(inp):
            actor_state, value_state = self.model.split_state(inp['state'])
            actor_inp = dict(
                obs=inp['obs'],
                mask=inp['mask'],
            )
            value_inp = dict(
                global_state=inp['global_state'],
                mask=inp['mask']
            )
            return {
                'actor_inp': actor_inp, 
                'actor_state': actor_state, 
                'value_inp': value_inp,
                'value_state': value_state,
            }

        if evaluation:
            inp = self.rms.process_obs_with_rms(inp)
        else:
            life_mask = inp.get('life_mask')
            inp = self.rms.process_obs_with_rms(inp, mask=life_mask)
        tf_inp = numpy2tensor(inp)
        tf_inp = split_input(tf_inp)

        return inp, tf_inp

    def _process_output(self, inp, out, evaluation):
        action, terms, state = out
        # convert to np.ndarray and restore the agent dimension
        b, a = inp['obs'].shape[:2]
        action, terms, prev_state = tensor2numpy((action, terms, inp['state']))
        prev_state = {k: v.reshape(b, a, v.shape[-1]) 
            for k, v in prev_state._asdict().items()}
        
        if not evaluation:
            terms.update({
                'obs': inp['obs'],
                'global_state': inp['global_state'],
                'mask': inp['mask'], 
                **prev_state,
            })
            if 'action_mask' in inp:
                terms['action_mask'] = inp['action_mask']
            if 'life_mask' in inp:
                terms['life_mask'] = inp['life_mask']

        return action, terms, state


def create_actor(config, model, name='mappo'):
    return MAPPOActor(config=config, model=model, name=name)
