from tools.tf_utils import numpy2tensor, tensor2numpy
from algo.zero.elements.actor import MAPPOActor

class MAPPOWithGoalActor(MAPPOActor):
    """ Calling Methods """
    def _split_input(self, inp: dict):
        actor_inp = dict(
            obs=inp['obs'],
            goal=inp['goal'],
        )
        value_inp = dict(
            global_state=inp['global_state'],
        )
        if 'action_mask' in inp:
            actor_inp['action_mask'] = inp['action_mask']
        if self.model.has_rnn:
            actor_inp['mask'] = inp['mask']
            value_inp['mask'] = inp['mask']
            actor_state, value_state = self.model.split_state(inp['state'])
            return {
                'actor_inp': actor_inp, 
                'value_inp': value_inp,
                'actor_state': actor_state, 
                'value_state': value_state,
            }
        else:
            return {
                'actor_inp': actor_inp, 
                'value_inp': value_inp,
            }


def create_actor(config, model, name='mappo'):
    return MAPPOWithGoalActor(config=config, model=model, name=name)
