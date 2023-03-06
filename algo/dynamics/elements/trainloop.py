from core.elements.trainloop import TrainingLoop as TrainingLoopBase
from tools.timer import Timer


class TrainingLoop(TrainingLoopBase):
    def _train(self):
        n = 0
        stats = {}
        if self.config.obs_normalization:
            self.norm_params_updated = False
        for _ in range(self.config.n_epochs):
            if self.config.ergodic:
                for data in self.buffer.ergodic_sample():
                    stats = self._train_with_data(data)
                    n += 1
                    if self.config.obs_normalization:
                        self.norm_params_updated = True
            else:
                data = self.sample_data()
                stats = self._train_with_data(data)
                n += 1
                if self.config.obs_normalization:
                    self.norm_params_updated = True
        
        return n, stats
    
    def _train_with_data(self, data):
        if data is None:
            return {}
        with Timer('train'):
            if self.config.obs_normalization:
                stats = self.trainer.train(data, update_norm_params=not self.norm_params_updated)
            else:
                stats = self.trainer.train(data)
        return stats