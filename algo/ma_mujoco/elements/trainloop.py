from core.elements.trainloop import TrainingLoop as TrainingLoopBase
from tools.timer import Timer


class TrainingLoop(TrainingLoopBase):
    def _train(self):
        for _ in range(self.trainer.config.n_epochs):
            data = self.sample_data()

            with Timer('train'):
                stats = self.trainer.train(data)
        
        return self.trainer.config.n_epochs, stats
