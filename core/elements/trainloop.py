from utility.timer import Timer
from utility.utils import config_attr


class TrainingLoopBase:
    def __init__(self, 
                 config, 
                 dataset, 
                 trainer, 
                 **kwargs):
        self.config = config_attr(self, config)
        self.dataset = dataset
        self.trainer = trainer

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