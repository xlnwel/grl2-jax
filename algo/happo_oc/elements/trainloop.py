from core.elements.trainloop import TrainingLoop as TrainingLoopBase
from tools.timer import Timer


class TrainingLoop(TrainingLoopBase):
    def imaginary_train(self, **kwargs):
        data = self.sample_data()

        return self.trainer.imaginary_train(data, **kwargs)

    def train(self, step, **kwargs):
        self._before_train(step)
        train_step, stats = self._train(**kwargs)
        self._after_train()

        return train_step, stats
    
    def _train(self, **kwargs):
        data = self.sample_data()
        if data is None:
            return 0, None

        with Timer('train'):
            stats = self.trainer.train(data, **kwargs)
        n = self.trainer.config.n_epochs * self.trainer.config.n_mbs
        
        return n, stats