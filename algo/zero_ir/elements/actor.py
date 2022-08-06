from core.elements.actor import Actor
from utility.tf_utils import numpy2tensor, tensor2numpy


class PPOActor(Actor):
    """ Calling Methods """
    def _process_input(self, inp: dict, evaluation: bool):
        if isinstance(inp, list):
            inp = inp[0]
        inp = self.rms.process_obs_with_rms(
            inp, mask=inp.get('life_mask'), 
            update_rms=self.config.get('update_obs_rms_at_execution', False)
        )
        # if 'prev_reward' in inp:
        #     inp['prev_reward'] = self.rms.process_reward_with_rms(
        #         inp['prev_reward'])
        tf_inp = numpy2tensor(inp)

        return inp, tf_inp


def create_actor(config, model, name='zero'):
    return PPOActor(config=config, model=model, name=name)
