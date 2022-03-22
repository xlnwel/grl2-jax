from core.elements.trainer import Trainer
from utility.timer import Timer
from utility.typing import AttrDict
from utility.utils import dict2AttrDict


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

    def train(self):
        train_step, stats = self._train()
        self._after_train()

        return train_step, stats

    def _train(self):
        raise NotImplementedError

    def _after_train(self):
        pass
