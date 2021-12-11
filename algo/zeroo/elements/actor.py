from algo.zero.elements.actor import PPOActor


class OPPOActor(PPOActor):
    def _process_output(self, inp, out, evaluation):
        action, terms, state = super()._process_output(inp, out, evaluation)
        if not evaluation:
            action = (*action, state[2])
            terms.update({
                **{k: inp[k] for k in self.rms.obs_names},
            })
        return action, terms, state

def create_actor(config, model, name='ppo'):
    return OPPOActor(config=config, model=model, name=name)
