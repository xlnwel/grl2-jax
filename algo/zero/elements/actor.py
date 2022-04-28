from algo.hm.elements.actor import MAPPOActor as MAPPOActorBase

class MAPPOActor(MAPPOActorBase):
    """ Calling Methods """
    def _split_input(self, inp: dict):
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
        if 'life_mask' in inp:
            value_inp['life_mask'] = inp['life_mask']
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


def create_actor(config, model, name='mappo'):
    return MAPPOActor(config=config, model=model, name=name)
