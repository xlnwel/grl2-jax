import collections
import logging

from core.elements.trainloop import TrainingLoopBase
from core.log import do_logging
from utility.tf_utils import numpy2tensor


logger = logging.getLogger(__name__)


class PPOTrainingLoop(TrainingLoopBase):
    def _train(self):
        train_step, stats = self._train_ppo()

        return train_step, stats

    def _train_ppo(self):
        stats = collections.defaultdict(list)

        for i in range(self.N_EPOCHS):
            for j in range(1, self.N_MBS+1):
                with self._sample_timer:
                    data = self._sample_data()

                data = numpy2tensor(data)

                with self._train_timer:
                    terms = self.trainer.train(**data)

                kl = terms.pop('kl').numpy()
                value = terms.pop('value').numpy()

                for k, v in terms.items():
                    stats[f'train/{k}'].append(v.numpy())
                if getattr(self.config, 'max_kl', None) and kl > self.config.max_kl:
                    break

                self._after_train_step()

                if self.config.value_update == 'reuse':
                    self.dataset.update('value', value)

            if getattr(self.config, 'max_kl', None) and kl > self.config.max_kl:
                do_logging(f'Eearly stopping after {i*self.N_MBS+j} update(s) '
                    f'due to reaching max kl. Current kl={kl:.3g}', logger=logger)
                break

            if self.config.value_update == 'once':
                self.dataset.update_value_with_func(self.compute_value)
            if self.config.value_update is not None:
                last_value = self.compute_value()
                self.dataset.finish(last_value)

            self._after_train_epoch()
        n = i * self.N_MBS + j

        for k, v in data.items():
            if isinstance(v, tuple):
                for kk, vv in v._asdict().items():
                    stats[f'train/{kk}'] = vv
        stats['train/kl'] = kl
        stats['train/value'] = value,
        stats['time/sample_mean'] = self._sample_timer.average()
        stats['time/train_mean'] = self._train_timer.average()
        stats['time/tps'] = 1 / self._train_timer.average()
        
        if self._train_timer.total() > 1000:
            self._train_timer.reset()

        return n, stats

    def _sample_data(self):
        return self.dataset.sample()

    def _after_train_step(self):
        """ Does something after each training step """
        pass

    def _after_train_epoch(self):
        """ Does something after each training epoch """
        pass
