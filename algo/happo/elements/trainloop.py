import jax

from core.typing import dict2AttrDict, tree_slice
from tools.timer import timeit
from algo.lka_common.elements.trainloop import TrainingLoop as TrainingLoopBase


class TrainingLoop(TrainingLoopBase):
    @timeit
    def _after_train(self, stats):
        mu_keys = self.model.policy_dist_stats('mu')
        data = dict2AttrDict({k: v 
            for k, v in self.training_data.items() 
            if k in ['obs', 'global_state', *mu_keys]
        })
        if self.model.has_rnn:
            data.state = tree_slice(self.training_data.state, indices=0, axis=1)
            data.state_reset = self.training_data.state_reset[:, :-1]
        if data is None:
            return stats
        
        self.rng, rng = jax.random.split(self.rng, 2)
        return self.compute_divs(rng, data, stats)
