from core.elements.actor import Actor as ActorBase
from tools.tf_utils import numpy2tensor


class Actor(ActorBase):
    """ Calling Methods """
    def _process_input(self, inp: dict, evaluation: bool):
        inp = self.rms.process_obs_with_rms(
            inp, mask=inp.get('life_mask'), 
            update_rms=self.config.get('update_obs_rms_at_execution', False)
        )
        # if 'prev_reward' in inp:
        #     inp['prev_reward'] = self.rms.process_reward_with_rms(
        #         inp['prev_reward'])
        tf_inp = numpy2tensor(inp)

        return inp, tf_inp
    
    def _process_output(
        self, 
        inp: dict, 
        out: tuple, 
        evaluation: bool
    ):
        action, terms, state = super()._process_output(inp, out, evaluation)
        if 'sid' in inp:
            terms['sid'] = inp['sid']
        if 'idx' in inp:
            terms['idx'] = inp['idx']
        if 'event' in inp:
            terms['event'] = inp['event']
        if 'hidden_state' in inp:
            terms['hidden_state'] = inp['hidden_state']
        
        return action, terms, state


def create_actor(config, model, name='zero'):
    return Actor(config=config, model=model, name=name)
