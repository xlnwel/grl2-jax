from core.elements.trainloop import TrainingLoop as TrainingLoopBase
from tools.utils import prefix_name
from tools.timer import Timer, timeit


class TrainingLoop(TrainingLoopBase):
    def _train(self):
        obs_rms = self.buffer.get_obs_rms()
        self.trainer.update_rms(obs_rms)

        n = 0
        stats = {}
        for _ in range(self.config.n_epochs):
            if self.config.ergodic:
                for data in self.buffer.ergodic_sample(n=self.config.training_data_size):
                    stats = self._train_with_data(data)
                    n += 1
            else:
                data = self.sample_data()
                if data is None:
                    break
                stats = self._train_with_data(data)
                n += 1

        return n, stats

    @timeit
    def valid_stats(self):
        data = self.buffer.range_sample(0, self.config.valid_data_size)
        data = self.trainer.process_data(data)

        _, stats = self.trainer.loss.loss(
            self.model.theta, 
            self.rng, 
            data
        )

        stats = prefix_name(stats, 'dynamics/valid')
        return stats
