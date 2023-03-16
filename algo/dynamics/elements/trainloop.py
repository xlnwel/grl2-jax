from core.elements.trainloop import TrainingLoop as TrainingLoopBase


class TrainingLoop(TrainingLoopBase):
    def _train(self):
        obs_rms = self.buffer.get_obs_rms()
        self.trainer.update_rms(obs_rms)

        n = 0
        stats = {}
        for _ in range(self.config.n_epochs):
            if self.config.ergodic:
                for data in self.buffer.ergodic_sample(n=self.config.samples_per_epochs):
                    stats = self._train_with_data(data)
                    n += 1
            else:
                data = self.sample_data()
                stats = self._train_with_data(data)
                n += 1
        
        return n, stats
