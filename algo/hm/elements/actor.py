from core.elements.actor import Actor
from utility.tf_utils import numpy2tensor, tensor2numpy


class MAPPOActor(Actor):
    """ Calling Methods """
    def _process_input(self, inp: dict, evaluation: bool):
        inp = self.rms.process_obs_with_rms(inp)
        inp['prev_reward'] = inp['prev_reward']
        tf_inp = numpy2tensor(inp)
        tf_inp = self._split_input(tf_inp)

        return inp, tf_inp

    def _split_input(self, inp):
        actor_inp = dict(
            obs=inp['obs'], 
            prev_reward=inp['prev_reward'], 
            prev_action=inp['prev_action'], 
        )
        value_inp = dict(
            global_state=inp['global_state'], 
            prev_reward=inp['prev_reward'], 
            prev_action=inp['prev_action'], 
        )
        if 'action_mask' in inp:
            actor_inp['action_mask'] = inp['action_mask']
        if self.model.has_rnn:
            for dest_inp in [actor_inp, value_inp]:
                dest_inp['mask'] = inp['mask']
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

    def _process_output(self, inp, out, evaluation):
        action, terms, state = out
        # convert to np.ndarray and restore the unit dimension
        b, u = inp['obs'].shape[:2]
        if self.model.has_rnn:
            action, terms, prev_state = tensor2numpy((action, terms, inp['state']))
            prev_state = {k: v.reshape(b, u, v.shape[-1]) 
                for k, v in prev_state._asdict().items()}
            
            if not evaluation:
                terms.update({
                    'mask': inp['mask'], 
                    **prev_state,
                })
        else:
            action, terms = tensor2numpy((action, terms))
        if not evaluation and self.rms is not None and self.rms.is_obs_normalized:
            terms.update({k: inp[k] for k in self.config.rms.obs_names})
        return action, terms, state


def create_actor(config, model, name='mappo'):
    return MAPPOActor(config=config, model=model, name=name)
