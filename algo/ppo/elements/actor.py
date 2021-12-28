from core.elements.actor import Actor


class PPOActor(Actor):
    """ Calling Methods """
    def _process_input(self, inp: dict, evaluation: bool):
        assert False, 'Reward RMS now take the first dimension as the sequential dimension. Modify code accordingly'
        inp = self.rms.process_obs_with_rms(inp, update_rms=not evaluation)
        return super()._process_input(inp, evaluation)

    def _process_output(self, inp, out, evaluation):
        out = super()._process_output(inp, out, evaluation)
        if not evaluation and self.rms.is_obs_normalized:
            out[1]['obs'] = inp['obs']
        return out


def create_actor(config, model, name='ppo'):
    return PPOActor(config=config, model=model, name=name)
