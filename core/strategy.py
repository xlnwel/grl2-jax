from typing import Union

from core.module import Model, ModelEnsemble, Trainer, TrainerEnsemble, Actor


def _set_attr(obj, name, attr):
    setattr(obj, name, attr)
    if isinstance(attr, dict):
        for k, v in attr.items():
            if not k.endswith(name):
                raise ValueError(f'Inconsistent error: {k} does not ends with {name}')
            setattr(obj, k, v)


class Strategy:
    """ Initialization """
    def __init__(self, 
                 *, 
                 model: Union[Model, ModelEnsemble], 
                 trainer: Union[Trainer, TrainerEnsemble], 
                 actor: Actor=None):
        _set_attr(self, 'model', model)
        _set_attr(self, 'trainer', trainer)
        self.actor = actor

    """ Checkpoint Ops """
    def restore(self):
        """ Restore model """
        if getattr(self, 'trainer', None) is not None:
            self.trainer.restore()
        elif getattr(self, 'model', None) is not None:
            self.model.restore()
        self.actor.restore_auxiliary_stats()

    def save(self, print_terminal_info=False):
        """ Save model """
        if getattr(self, 'trainer', None) is not None:
            self.trainer.save(print_terminal_info)
        elif getattr(self, 'model', None) is not None:
            self.model.save(print_terminal_info)
        self.actor.save_auxiliary_stats()
