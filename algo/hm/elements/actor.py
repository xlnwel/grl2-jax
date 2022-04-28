from core.elements.actor import Actor
from utility.tf_utils import numpy2tensor, tensor2numpy


class MAPPOActor(Actor):
    """ Calling Methods """
    def _process_input(self, inp: dict, evaluation: bool):
        inp = self.rms.process_obs_with_rms(inp, mask=inp.get('life_mask'))
        if 'prev_reward' in inp:
            inp['prev_reward'] = self.rms.process_reward_with_rms(inp['prev_reward'])
        tf_inp = numpy2tensor(inp)
        tf_inp = self._split_input(tf_inp)

        return inp, tf_inp

    def _split_input(self, inp: dict):
        actor_inp = dict(
            obs=inp['obs'], 
        )
        global_state = inp['global_state'] if 'global_state' in inp else inp['obs']
        value_inp = dict(
            global_state=global_state, 
        )
        if 'action_mask' in inp:
            actor_inp['action_mask'] = inp['action_mask']
        if self.model.has_rnn:
            for dest_inp in [actor_inp, value_inp]:
                dest_inp['mask'] = inp['mask']
                if 'prev_reward' in inp:
                    dest_inp['prev_reward'] = inp['prev_reward']
                if 'prev_action' in inp:
                    dest_inp['prev_action'] = inp['prev_action']
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
            action, terms, prev_state = tensor2numpy(
                (action, terms, inp['state']))

            if not evaluation:
                actor_state, value_state = self.model.split_state(prev_state)
                terms['mask'] = inp['mask']
                if actor_state is not None:
                    terms.update({
                        f'actor_{k}': v.reshape(b, u, v.shape[-1]) 
                        for k, v in actor_state._asdict().items()
                    })
                if value_state is not None:
                    terms.update({
                        f'value_{k}': v.reshape(b, u, v.shape[-1]) 
                        for k, v in value_state._asdict().items()
                    })
        else:
            action, terms = tensor2numpy((action, terms))
        if not evaluation and self.rms is not None and self.rms.is_obs_normalized:
            terms.update({k: inp[k] for k in self.config.rms.obs_names})
        return action, terms, state


def create_actor(config, model, name='mappo'):
    return MAPPOActor(config=config, model=model, name=name)
