from core.elements.actor import Actor


class PPOActor(Actor):
    """ Calling Methods """
    def _process_input(self, inp: dict, evaluation: bool):
        inp = self.rms.process_obs_with_rms(inp, update_rms=False)
        inp, tf_inp = super()._process_input(inp, evaluation)
        state = tf_inp.pop('state')
        tf_inp.update({
            **state._asdict()
        })
        return inp, tf_inp

    def _process_output(self, inp, out, evaluation):
        action, terms, state = super()._process_output(inp, out, evaluation)
        if not evaluation:
            action = (*action, state[2])
            terms.update({
                **{k: inp[k] for k in self.rms.obs_names},
            })
        return action, terms, state


def create_actor(config, model, name='ppo'):
    return PPOActor(config=config, model=model, name=name)
