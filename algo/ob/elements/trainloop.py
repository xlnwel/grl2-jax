import logging
import numpy as np

from core.elements.trainloop import TrainingLoopBase
from core.log import do_logging
from utility.tf_utils import numpy2tensor, tensor2numpy


logger = logging.getLogger(__name__)


class PPOTrainingLoop(TrainingLoopBase):
    def _train(self):
        train_step, stats = self._train_ppo()

        return train_step, stats

    def _train_ppo(self):
        def train():
            raw_data = None
            for i in range(self.config.n_epochs):
                for j in range(1, self.config.n_mbs+1):
                    if self.use_dataset:
                        with self._sample_timer:
                            data = self._sample_data()
                        if data is None:
                            return
                    else:
                        with self._sample_timer:
                            raw_data = self._sample_data()
                            if raw_data is None:
                                return
                            data = numpy2tensor(raw_data)

                    with self._train_timer:
                        terms = self.trainer.train(**data)

                    kl = terms.pop('kl').numpy()
                    # value = terms.pop('value').numpy()

                    if getattr(self.config, 'max_kl', None) and kl > self.config.max_kl:
                        break

                    self._after_train_step()

                    # if self.config.value_update == 'reuse':
                    #     self.dataset.update('value', value)

                if getattr(self.config, 'max_kl', None) and kl > self.config.max_kl:
                    do_logging(f'Eearly stopping after {i*self.config.n_mbs+j} update(s) '
                        f'due to reaching max kl. Current kl={kl:.3g}', logger=logger)
                    break

                # if self.config.value_update == 'once':
                #     self.dataset.update_value_with_func(self.compute_value)
                # if self.config.value_update is not None:
                #     last_value = self.compute_value()
                #     self.dataset.finish(last_value)

                self._after_train_epoch()
            n = i * self.config.n_mbs + j

            stats = {'train/kl': kl}
            if raw_data is None:
                raw_data = tensor2numpy(data)

            return n, stats, raw_data, tensor2numpy(terms)

        def combine_stats(stats, data, terms, maxlen=100):
            size = next(iter(data.values())).shape[0]
            # we only sample a small amount of data to reduce the cost
            idx = np.random.randint(0, size, maxlen)

            for k, v in data.items():
                if isinstance(v, tuple):
                    for kk, vv in v._asdict().items():
                        vv_shape = np.shape(vv)
                        stats[f'train/{kk}'] = vv[idx] \
                            if vv_shape != () and vv_shape[0] == size else vv
                else:
                    v_shape = np.shape(v)
                    stats[f'train/{k}'] = v[idx] \
                        if v_shape != () and v_shape[0] == size else v

            stats.update(
                **{f'train/{k}': v[idx] 
                    if np.shape(v) != () and np.shape(v)[0] == size else v 
                    for k, v in terms.items()}, 
                **{f'time/{t.name}_total': t.total() 
                    for t in [self._sample_timer, self._train_timer]},
                **{f'time/{t.name}': t.average() 
                    for t in [self._sample_timer, self._train_timer]},
            )
            return stats

        result = train()
        if result is None:
            return 0, None
        n, stats, data, terms = result
        stats = combine_stats(stats, data, terms)

        if self._train_timer.total() > 1000:
            self._train_timer.reset()
            self._sample_timer.reset()

        return n, stats

    def _sample_data(self):
        return self.dataset.sample()

    def _after_train_step(self):
        """ Does something after each training step """
        pass

    def _after_train_epoch(self):
        """ Does something after each training epoch """
        pass
