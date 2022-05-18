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
        def get_data():
            raw_data = None
            if self.use_dataset:
                with self._sample_timer:
                    data = self._sample_data()
            else:
                with self._sample_timer:
                    raw_data = self._sample_data()
                    data = None if raw_data is None else numpy2tensor(raw_data)
            return raw_data, data

        def combine_stats(stats, data, terms, max_record_size=100):
            batch_size = next(iter(data.values())).shape[0]
            # we only sample a small amount of data to reduce the cost
            idx = np.random.randint(0, batch_size, max_record_size)

            for k, v in data.items():
                if isinstance(v, tuple):
                    for kk, vv in v._asdict().items():
                        vv_shape = np.shape(vv)
                        stats[f'train/{kk}'] = vv[idx] \
                            if vv_shape != () and vv_shape[0] == batch_size else vv
                else:
                    v_shape = np.shape(v)
                    stats[f'train/{k}'] = v[idx] \
                        if v_shape != () and v_shape[0] == batch_size else v

            stats.update(
                **{f'train/{k}': v[idx] 
                    if np.shape(v) != () and np.shape(v)[0] == batch_size else v 
                    for k, v in terms.items()}, 
                **{f'time/{t.name}_total': t.total() 
                    for t in [self._sample_timer, self._train_timer]},
                **{f'time/{t.name}': t.average() 
                    for t in [self._sample_timer, self._train_timer]},
            )
            return stats

        def train(max_record_size=100):
            for i in range(self.config.n_epochs):
                for j in range(1, self.config.n_mbs+1):
                    raw_data, data = get_data()
                    if data is None:
                        return

                    with self._train_timer:
                        terms = self.trainer.train(**data)
                    
                    if self.config.get('debug', False):
                        print(i * self.config.n_mbs + j)
                        logprob = data['logprob'].numpy()
                        print('logprob', logprob.mean(), logprob.max(), logprob.min())
                        kl = terms.pop('approx_kl').numpy()
                        print('approx kl', kl)
                        print('kl', np.mean(terms.pop('kl').numpy()))
                        print('pg_loss', terms.pop('pg_loss').numpy())
                        print('gpo_loss', terms.pop('gpo_loss').numpy())
                        print('actor_norm', terms.pop('actor_norm').numpy())
                        print('entropy', terms.pop('entropy').numpy())
                        print('new_prob', terms.pop('new_prob').numpy())
                        logits = terms.pop('logits').numpy()
                        print('logits', logits.mean(), logits.max(), logits.min())
                        import tensorflow as tf
                        probs = tf.nn.softmax(logits).numpy()
                        print('probs', probs.mean(), probs.max(), probs.min())
                    else:
                        kl = terms.pop('approx_kl').numpy()
                    # print('actor_var_norm', terms.pop('actor_var_norm').numpy())
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

                if self.config.value_update == 'once':
                    self.dataset.update_value_with_func(self.compute_value)
                if self.config.value_update is not None:
                    last_value = self.compute_value()
                    self.dataset.finish(last_value)

                self._after_train_epoch()
            n = i * self.config.n_mbs + j

            if raw_data is None:
                raw_data = tensor2numpy(data)
            stats = {'train/kl': kl}
            stats = combine_stats(
                stats, 
                raw_data, 
                tensor2numpy(terms), 
                max_record_size=max_record_size
            )

            return n, stats

        result = train()
        if result is None:
            return 0, None
        n, stats = result

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
