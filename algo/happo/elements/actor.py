from algo.ma_common.elements.actor import Actor as ActorBase


class Actor(ActorBase):
    def compute_value(self, inps):
        inps = self._process_input(inps, False)
        value = self.model.compute_value(inps)
        return value

def create_actor(config, model, name='actor'):
    return Actor(config=config, model=model, name=name)
