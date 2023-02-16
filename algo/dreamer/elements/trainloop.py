from core.elements.trainloop import TrainingLoop as TrainingLoopBase
from tools.timer import Timer


class TrainingLoop(TrainingLoopBase):
    def _train(self):
        for _ in range(self.trainer.config.n_epochs):
            data = self.sample_data()

            with Timer('train'):
                stats = self.trainer.train(data)
        
        return self.trainer.config.n_epochs, stats

    def model_train(self, step, **kwargs):
        self._before_train(step)
        train_step, stats = self._model_train(**kwargs)
        self._after_train()

        return train_step, stats

    def _model_train(self):
        for _ in range(self.trainer.config.n_epochs):
            data = self.sample_data()

            with Timer('train'):
                stats = self.trainer.model_train(data)
        
        return self.trainer.config.n_epochs, stats
