from core.elements.actor import Actor as ActorBase, create_actor

class Actor(ActorBase):
    def __call__(self, inp, evaluation):
        out = self.model.action(inp, evaluation)
        out = self._process_output(inp, out, evaluation)
        return out

    def _process_output(self, inp, out, evaluation):
        return out

def create_actor(config, model, name='actor'):
    return Actor(config=config, model=model, name=name)
