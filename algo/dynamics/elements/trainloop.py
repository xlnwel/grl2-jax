from core.elements.trainloop import TrainingLoop as TrainingLoopBase
from tools.utils import prefix_name


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
                if data is None:
                    break
                stats = self._train_with_data(data)
                n += 1

        if n > 0:
            valid_stats = self.valid_stats()
            stats.update(valid_stats)

        return n, stats

    def valid_stats(self):
        data = self.buffer.range_sample(0, self.config.n_valid_samples)
        data = self.trainer.process_data(data)

        _, stats = self.trainer.loss.loss(
            self.model.theta, 
            self.rng, 
            data
        )

        stats = prefix_name(stats, 'dynamics/valid')
        return stats