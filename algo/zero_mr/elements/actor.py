from core.elements.actor import Actor as ActorBase


class Actor(ActorBase):
    """ Calling Methods """
    def _process_input(self, inp: dict, evaluation: bool):
        inp = self.rms.process_obs_with_rms(
            inp, mask=inp.get('life_mask'), 
            update_rms=self.config.get('update_obs_rms_at_execution', False)
        )

        return inp
    
    def _process_output(
        self, 
        inp: dict, 
        out: tuple, 
        evaluation: bool
    ):
        action, stats, state = super()._process_output(inp, out, evaluation)
        if 'sid' in inp:
            stats['sid'] = inp['sid']
        if 'idx' in inp:
            stats['idx'] = inp['idx']
        if 'event' in inp:
            stats['event'] = inp['event']
        if 'hidden_state' in inp:
            stats['hidden_state'] = inp['hidden_state']
        
        return action, stats, state


def create_actor(config, model, name='zero'):
    return Actor(config=config, model=model, name=name)
