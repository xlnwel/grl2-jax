import jax

from tools.timer import timeit
from algo.lka_common.elements.trainloop import TrainingLoop as TrainingLoopBase


class TrainingLoop(TrainingLoopBase):
    @timeit
    def _after_train(self, stats):
        BATCH_SIZE = 400
        mu_keys = self.model.policy_dist_stats('mu')
        data = self.buffer.sample_from_recency(
            batch_size=BATCH_SIZE, 
            sample_keys=['obs', *mu_keys], 
            n=BATCH_SIZE
        )
        if data is None:
            return stats

        self.rng, rng = jax.random.split(self.rng, 2)
        return self.compute_divs(rng, data, stats)
