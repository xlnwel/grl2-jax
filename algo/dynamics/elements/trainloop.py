import collections
import numpy as np

from core.typing import AttrDict
from core.elements.trainloop import TrainingLoop as TrainingLoopBase
from tools.display import print_dict_info
from tools.utils import prefix_name
from tools.timer import timeit


class TrainingLoop(TrainingLoopBase):
    def post_init(self):
        self.emodel_metrics = collections.deque(maxlen=self.config.n_epochs)

    def _before_train(self, step):
        obs_rms = self.buffer.get_obs_rms()
        self.model.update_obs_rms(obs_rms)

    def _train(self):
        n = 0
        stats = {}
        if self.config.ergodic:
            for data in self.buffer.ergodic_sample(n=self.config.training_data_size):
                stats = self._train_with_data(data)
                n += 1
        else:
            data = self.sample_data()
            if data is None:
                return 0, AttrDict()
            stats = self._train_with_data(data)
            if self.buffer.type() == 'per':
                priority = stats['dynamics/priority']
                self.buffer.update_priorities(priority, data.idxes)
            n += 1
            self.emodel_metrics.append(stats.pop('dynamics/emodel_metrics'))

        return n, stats

    def _after_train(self, stats):
        self.model.rank_elites(np.mean(self.emodel_metrics, 0))
        return stats

    @timeit
    def valid_stats(self):
        data = self.buffer.sample_from_recency(self.config.valid_data_size, add_seq_dim=True)
        data = self.trainer.process_data(data)

        _, stats = self.trainer.loss.loss(
            self.model.theta, 
            self.rng, 
            data
        )

        stats = prefix_name(stats, 'dynamics/valid')
        return stats
