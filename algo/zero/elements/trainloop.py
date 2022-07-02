import logging
import numpy as np

from core.elements.trainloop import TrainingLoopBase
from core.log import do_logging
from utility import div
from utility.display import print_dict_info
from utility.tf_utils import numpy2tensor, tensor2numpy


logger = logging.getLogger(__name__)


def _get_pi(data, is_action_discrete):
    if is_action_discrete:
        pi = data['mu']
    else:
        pi = (data['mu_mean'], data['mu_std'])
    return pi


class TrainingLoop(TrainingLoopBase):
    def _post_init(self):
        super()._post_init()
        self._prev_pi = None
        self._step = 0

    def _before_train(self, step):
        self._train_step = step

    def _train(self):
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

        def combine_stats(stats, data, terms, max_record_size=10):
            batch_size = next(iter(data.values())).shape[0]
            # we only sample a small amount of data to reduce the cost
            if max_record_size is not None and max_record_size < batch_size:
                idx = np.random.randint(0, batch_size, max_record_size)
            else:
                idx = np.arange(batch_size)

            for k, v in data.items():
                if isinstance(v, tuple):
                    for kk, vv in v._asdict().items():
                        vv_shape = np.shape(vv)
                        stats[f'data/{kk}'] = vv[idx] \
                            if vv_shape != () and vv_shape[0] == batch_size else vv
                else:
                    v_shape = np.shape(v)
                    stats[f'data/{k}'] = v[idx] \
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

        def train(max_record_size=10):
            raw_data, data = get_data()
            if data is None:
                return

            if self.config.inner_steps is not None and self._step == self.config.inner_steps:
                with self._train_timer:
                    terms = self.trainer.meta_train(**data)
                self._step = 0
                if raw_data is None:
                    raw_data = tensor2numpy(data)
                pi = _get_pi(raw_data, self.trainer.env_stats.is_action_discrete)
                if self._prev_pi is not None:
                    kl = div.kl(pi, self._prev_pi, self.trainer.env_stats.is_action_discrete)
                    stats = {'train/kl': kl}
                else:
                    stats = {}
                self._prev_pi = pi
                stats = combine_stats(
                    stats, 
                    raw_data, 
                    tensor2numpy(terms), 
                    max_record_size=max_record_size
                )
            else:
                with self._train_timer:
                    terms = self.trainer.train(**data)
                self._step += 1
                stats = {}

            return 1, stats

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
