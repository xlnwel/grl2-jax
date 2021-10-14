import tensorflow as tf

from core.elements.model import Model, Ensemble
from utility.utils import config_attr


class Loss(tf.Module):
    """ We do not need this class for now. However, 
    in case one day we need multiple GPUs for training, 
    this class will come in handy.
    """
    def __init__(self,
                 *,
                 config: dict,
                 model: Model,
                 use_tf: bool=False,
                 name):
        super().__init__(name=f'{name}_loss')
        config = config.copy()
        config_attr(self, config, filter_dict=True)

        self.model = model
        [setattr(self, k, v) for k, v in self.model.items()]
        if use_tf:
            self.loss = tf.function(self.loss)

    def loss(self):
        raise NotImplementedError


class LossEnsemble(Ensemble):
    pass
