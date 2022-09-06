import logging
import jax
import jax.numpy as jnp

from core.elements.trainloop import TrainingLoopBase
from core.log import do_logging
from core.typing import AttrDict
from tools.display import print_dict_info
from tools.timer import Timer
from .utils import compute_inner_steps


logger = logging.getLogger(__name__)


class TrainingLoop(TrainingLoopBase):
    def _post_init(self):
        super()._post_init()
        self.config = compute_inner_steps(self.config)
        self.config.debug = True
        self.rl_params = self.trainer.get_rl_weights()
        self._use_meta = bool(self.config.inner_steps)
        self._meta_train_timer = Timer('meta_train')
        self._step = 0

    def _before_train(self, step):
        self._step += 1

    def _train(self):
        def get_data():
            with self._sample_timer:
                data = self._sample_data()
            data.setdefault('global_state', data.obs)
            if 'next_obs' in data:
                data.setdefault('next_global_state', data.next_obs)
            return data

        def combine_stats(data, stats, max_record_size=10, axis=0):
            batch_size = next(iter(data.values())).shape[axis]
            # we only sample a small amount of data to reduce the cost
            if max_record_size is not None and max_record_size < batch_size:
                idx = jax.random.randint(
                    self.model.rng, 
                    (max_record_size,), 
                    minval=0, 
                    maxval=batch_size
                )
            else:
                idx = jnp.arange(batch_size)

            stats = {k if '/' in k else f'train/{k}': jnp.take(v, idx, axis) 
                if isinstance(v, jnp.DeviceArray) and v.ndim > axis else v 
                for k, v in stats.items()}
            stats.update({f'data/{k}': jnp.take(v, idx, axis=axis)
                for k, v in data.items() if v is not None})
            stats.update(
                **{f'time/{t.name}_total': t.total() 
                    for t in [self._sample_timer, self._train_timer, 
                        self._meta_train_timer]},
                **{f'time/{t.name}': t.average() 
                    for t in [self._sample_timer, self._train_timer, 
                        self._meta_train_timer]},
            )
            return stats

        def train(max_record_size=10):
            data = get_data()
            assert isinstance(data, AttrDict), type(data)
            if data is None:
                self._step -= 1
                return
            do_meta_step = self._use_meta and \
                self._step % (self.config.inner_steps + self.config.extra_meta_step) == 0
            # print(self._step, self.config.inner_steps, self.config.extra_meta_step)
            # print(do_meta_step)

            if do_meta_step:
                self.trainer.set_rl_weights(self.rl_params)
                with self._meta_train_timer:
                    eta, theta, self.trainer.params, stats = \
                        self.trainer.meta_train(
                            self.model.eta, 
                            self.model.theta, 
                            self.trainer.params, 
                            data
                        )
                self.model.set_weights(theta)
                self.model.set_weights(eta)
                self.rl_params = self.trainer.get_rl_weights()
            else:
                use_meta = self._use_meta
                with self._train_timer:
                    theta, self.trainer.params.theta, stats = \
                        self.trainer.train(
                            self.model.theta, 
                            self.model.eta, 
                            self.trainer.params.theta, 
                            data, 
                            use_meta=use_meta, 
                        )
                self.model.set_weights(theta)

            if not self._use_meta or do_meta_step:
                with Timer('combine stats', period=1000):
                    stats = combine_stats(
                        data, 
                        stats, 
                        max_record_size=max_record_size, 
                        axis=int(self._use_meta)
                    )
            else:
                stats = {}

            return 1, stats

        result = train()
        if result is None:
            return 0, None
        n, stats = result

        if self._train_timer.total() > 1000:
            self._train_timer.reset()
            self._meta_train_timer.reset()
            self._sample_timer.reset()
        return n, stats

    def _rl_vars(self):
        return self.trainer.model['rl'].policy.variables \
            + self.trainer.model['rl'].value.variables

    def _meta_vars(self):
        return self.trainer.model['meta'].policy.variables \
            + self.trainer.model['meta'].value.variables

    def _sample_data(self):
        return self.dataset.sample()

    def _after_train_step(self):
        """ Does something after each training step """
        pass

    def _after_train_epoch(self):
        """ Does something after each training epoch """
        pass
