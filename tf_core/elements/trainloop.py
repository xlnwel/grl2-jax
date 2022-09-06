from core.elements.trainer import Trainer
from tools.timer import Timer
from core.typing import AttrDict
from tools.utils import dict2AttrDict


class TrainingLoopBase:
    def __init__(
        self, 
        config: AttrDict, 
        dataset, 
        trainer: Trainer, 
        **kwargs
    ):
        self.config = dict2AttrDict(config)
        self.dataset = dataset
        self.trainer = trainer

        self.use_dataset = self.config.get('use_dataset', False)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._sample_timer = Timer('sample')
        self._train_timer = Timer('train')
        self._post_init()

    def _post_init(self):
        pass

    def train(self, step):
        self._before_train(step)
        train_step, stats = self._train()
        self._after_train()

        return train_step, stats

    def _train(self):
        raise NotImplementedError

    def _before_train(self, step):
        pass

    def _after_train(self):
        pass
