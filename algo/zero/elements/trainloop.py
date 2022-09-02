import logging
import numpy as np

from core.elements.trainloop import TrainingLoopBase
from core.log import do_logging
from utility.tf_utils import numpy2tensor, tensor2numpy
from utility.timer import Timer
from .utils import compute_inner_steps


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
        self.config = compute_inner_steps(self.config)
        self._use_meta = bool(self.config.inner_steps)
        self._meta_train_timer = Timer('meta_train')
        self._sync_timer = Timer('sync_train')
        self._prev_data = []
        self._new_iter = True
        self.config.debug = True
        self._step = 0

    def _before_train(self, step):
        self._step += 1

    def _train(self):
        def get_data():
            raw_data = None
            if self.use_dataset:
                with self._sample_timer:
                    data = self._sample_data()
            else:
                with self._sample_timer:
                    raw_data = self._sample_data()
                    data = numpy2tensor(raw_data)
            
            return raw_data, data

        def combine_stats(stats, data, terms, max_record_size=10, axis=0):
            batch_size = next(iter(data.values())).shape[axis]
            # we only sample a small amount of data to reduce the cost
            if max_record_size is not None and max_record_size < batch_size:
                idx = np.random.randint(0, batch_size, max_record_size)
            else:
                idx = np.arange(batch_size)

            for k, v in data.items():
                if isinstance(v, tuple):
                    for kk, vv in v._asdict().items():
                        vv_shape = np.shape(vv)
                        stats[f'data/{kk}'] = np.take(vv, idx, axis) \
                            if len(vv_shape) > axis and vv_shape[axis] == batch_size else vv
                else:
                    v_shape = np.shape(v)
                    stats[f'data/{k}'] = np.take(v, idx, axis) \
                        if len(v_shape) > axis and v_shape[axis] == batch_size else v

            stats.update(
                **{k if '/' in k else f'train/{k}': np.take(v, idx, axis) 
                    if len(np.shape(v)) > axis and np.shape(v)[axis] == batch_size else v 
                    for k, v in terms.items()}, 
                **{f'time/{t.name}_total': t.total() 
                    for t in [self._sample_timer, self._train_timer, 
                        self._meta_train_timer, self._sync_timer]},
                **{f'time/{t.name}': t.average() 
                    for t in [self._sample_timer, self._train_timer, 
                        self._meta_train_timer, self._sync_timer]},
            )
            return stats

        def train(max_record_size=10):
            raw_data, data = get_data()
            if data is None:
                self._step -= 1
                return
            do_meta_step = self._use_meta and \
                self._step % (self.config.inner_steps + self.config.extra_meta_step) == 0
            # print(self._step, self.config.inner_steps, self.config.extra_meta_step)
            # print(do_meta_step)
            # print_dict_info(data)
            
            if do_meta_step:
                if self.config.get('debug', False):
                    # test data consistency
                    assert len(self._prev_data) in (
                        self.config.inner_steps, self.config.inner_steps - 1), len(self._prev_data)
                    for i, pd in enumerate(self._prev_data):
                        for k, d1, d2 in zip(pd.keys(), pd.values(), data.values()):
                            assert len(d2) == self.config.inner_steps + self.config.extra_meta_step, (k, len(d2))
                            np.testing.assert_allclose(d1, d2[i])
                    self._prev_data = []
                    # test consistency of rl optimizer's variables
                    rl_vars = self._prev_rl_opt_vars
                    meta_vars = self.trainer.optimizers['meta_rl'].opt_variables
                    for v1, v2 in zip(rl_vars, meta_vars):
                        np.testing.assert_allclose(v1, v2.numpy())
                    # test consistency of rl variables
                    rl_vars = self._prev_rl_vars
                    meta_vars = self._meta_vars()
                    for v1, v2 in zip(rl_vars, meta_vars):
                        np.testing.assert_allclose(v1, v2.numpy())
                with self._meta_train_timer:
                    terms = self.trainer.meta_train(**data)
                with self._sync_timer:
                    if self.config.inner_steps == 1 and self.config.extra_meta_step == 0:
                        pass
                    elif self.config.extra_meta_step == 0:
                        self.trainer.sync_nets(forward=True)
                    elif self.trainer.config.L == 0:
                         self.trainer.sync_nets(forward=None)
                    else:
                        self.trainer.sync_nets(forward=False)
                if self.config.get('debug', False):
                    # test consistency of variables stored by optimizers
                    rl_vars = self.trainer.optimizers['rl'].variables
                    meta_vars = self.trainer.optimizers['meta_rl'].variables
                    for v1, v2 in zip(rl_vars, meta_vars):
                        np.testing.assert_allclose(v1.numpy(), v2.numpy(), err_msg=v1.name)
                    # test consistency of rl optimizer's variables
                    rl_vars = self.trainer.optimizers['rl'].opt_variables
                    meta_vars = self.trainer.optimizers['meta_rl'].opt_variables
                    for v1, v2 in zip(rl_vars, meta_vars):
                        np.testing.assert_allclose(v1.numpy(), v2.numpy(), err_msg=v1.name)
                    # test consistency of variables
                    rl_vars = self._rl_vars()
                    meta_vars = self._meta_vars()
                    for v1, v2 in zip(rl_vars, meta_vars):
                        np.testing.assert_allclose(v1, v2.numpy())
                    self._new_iter = True
                    print('meta train step')
            else:
                if self.config.get('debug', False):
                    self._prev_data.append(data)
                    if self._new_iter:
                        # test consistency of rl optimizer's variables
                        rl_vars = self.trainer.optimizers['rl'].opt_variables
                        meta_vars = self.trainer.optimizers['meta_rl'].opt_variables
                        self._prev_rl_opt_vars = [v.numpy() for v in rl_vars]
                        for v1, v2 in zip(rl_vars, meta_vars):
                            np.testing.assert_allclose(v1.numpy(), v2.numpy(), err_msg=v1.name)
                        # test consistency of variables
                        rl_vars = self._rl_vars()
                        self._prev_rl_vars = [v.numpy() for v in rl_vars]
                        meta_vars = self._meta_vars()
                        for v1, v2 in zip(rl_vars, meta_vars):
                            np.testing.assert_allclose(v1.numpy(), v2.numpy(), err_msg=v1.name)
                if not self._use_meta:
                    use_meta = numpy2tensor(False)
                elif self.trainer.config.meta_type == 'bmg':
                    use_meta = (self._step + 1) % (self.config.inner_steps + self.config.extra_meta_step) != 0
                    use_meta = numpy2tensor(use_meta)
                elif self.trainer.config.meta_type == 'plain':
                    use_meta = numpy2tensor(True)
                else:
                    raise NotImplementedError
                with self._train_timer:
                    terms = self.trainer.train(**data, use_meta=use_meta)
                if self.config.get('debug', False):
                    self._prev_grads_norm = terms['grads_norm']
                    print('raw train step')
                    self._new_iter = False

            stats = {}
            if not self._use_meta or do_meta_step:
                if raw_data is None:
                    raw_data = tensor2numpy(data)
                stats = combine_stats(
                    stats, 
                    raw_data, 
                    tensor2numpy(terms), 
                    max_record_size=max_record_size, 
                    axis=int(self._use_meta)
                )

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
