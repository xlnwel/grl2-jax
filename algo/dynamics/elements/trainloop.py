from core.elements.trainloop import TrainingLoop as TrainingLoopBase


class TrainingLoop(TrainingLoopBase):
    def _train(self):
        n = 0
        stats = {}
        for _ in range(self.config.n_epochs):
            if self.config.ergodic:
                for data in self.buffer.ergodic_sample():
                    stats = self._train_with_data(data)
                    n += 1
            else:
                data = self.sample_data()
                stats = self._train_with_data(data)
                n += 1
        
        return n, stats
