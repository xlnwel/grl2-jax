import tensorflow as tf

from core.elements.model import Model, Ensemble, ModelEnsemble
from utility.typing import AttrDict


class Loss(tf.Module):
    def __init__(
        self,
        *,
        config: AttrDict,
        model: Model,
        use_tf: bool=False,
        name: str
    ):
        super().__init__(name=f'{name}_loss')
        self.config = config

        self.model = model
        [setattr(self, k, v) for k, v in self.model.items()]
        if use_tf:
            self.loss = tf.function(self.loss)
        self._post_init()

    def loss(self):
        raise NotImplementedError

    def _post_init(self):
        """ Add some additional attributes and do some post processing here """
        pass

    def log_for_debug(self, tape, terms, debug=True, **data):
        if debug and self.config.get('debug', True):
            with tape.stop_recording():
                terms.update(data)


class LossEnsemble(Ensemble):
    def __init__(
        self, 
        *, 
        config, 
        model: ModelEnsemble,
        constructor, 
        name, 
        **classes
    ):
        super().__init__(
            config=config,
            constructor=constructor,
            name=name,
            has_ckpt=False, 
            **classes
        )

        self.model = model

    def _init_components(self, constructor, classes):
        if classes:
            component_configs = [k for k, v in self.config.items() 
                if isinstance(v, dict)]
            if set(component_configs) != set(classes):
                raise ValueError(
                    f'Inconsistent configs and classes: '
                    f'config({component_configs}) != classes({classes})'
                )
            self.components = {}
            for k, cls in classes.items():
                obj = constructor(self.config[k], cls, k)
                self.components[k] = obj
                setattr(self, k, obj)
        else:
            self.components = constructor(self.config)
            [setattr(self, n, m) for n, m in self.components.items()]
