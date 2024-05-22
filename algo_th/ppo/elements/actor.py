from th.core.elements.actor import Actor as ActorBase


class Actor(ActorBase):
  def compute_value(self, inps):
    inps = self.process_obs_with_rms(inps, False)
    value = self.model.compute_value(inps)
    return value


def create_actor(config, model, name='actor'):
  return Actor(config=config, model=model, name=name)
