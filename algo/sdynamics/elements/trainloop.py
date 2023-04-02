from core.elements.trainloop import TrainingLoop as TrainingLoopBase
from tools.utils import prefix_name


class TrainingLoop(TrainingLoopBase):
    def _train(self):
        env_state_rms = self.buffer.get_obs_rms() # Here get_obs_rms is defined outside sdynamics, and we don't change it.
        self.trainer.update_rms(env_state_rms)

        n = 0
        stats = {}
        for _ in range(self.config.n_epochs):
            if self.config.ergodic:
                for data in self.buffer.ergodic_sample(n=self.config.training_data_size):
                    stats = self._train_with_data(data)
                    n += 1
            else:
                data = self.sample_data()
                stats = self._train_with_data(data)
                n += 1
        
        if n > 0:
            valid_stats = self.valid_stats()
            stats.update(valid_stats)

        return n, stats

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