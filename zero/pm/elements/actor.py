import numpy as np

from core.elements.actor import Actor
from tools.display import print_dict


class MAPPOActor(Actor):
    def _process_output(self, inp, out, evaluation):
        action, terms, state = super()._process_output(inp, out, evaluation)
        plogits = terms.pop('plogits', None)
        pentropy = terms.pop('pentropy', None)
        paction = terms.pop('paction', None)
        entropy = terms.pop('entropy', None)

        if np.any(np.logical_or(action >= self.model.policy.action_dim, action < 0)):
            print_dict(terms, 'terms')
            print('plogits', plogits)
            print('pentropy', pentropy)
            print('paction', paction)
            print('action', action)
            print('entropy', entropy)
            exit()
        return action, terms, state

def create_actor(config, model, name='actor'):
    return MAPPOActor(config=config, model=model, name=name)
