import collections
import functools
import logging
from typing import Dict
import numpy as np
import tensorflow as tf

from core.elements.strategy import Strategy, create_strategy
from core.mixin.strategy import TrainingLoopBase
from core.log import do_logging


logger = logging.getLogger(__name__)


class PPOTrainingLoop(TrainingLoopBase):
    def _post_init(self):
        value_state_keys = self.trainer.model.state_keys
        self._value_sample_keys = [
            'global_state', 'value', 'traj_ret', 'mask'
        ] + list(value_state_keys)

    def _train(self):
        train_step, stats = self._train_ppo()
        extra_stats = self._train_extra_vf()
        stats.update(extra_stats)

        return train_step, stats

    def _train_ppo(self):
        stats = collections.defaultdict(list)

        for i in range(self.N_EPOCHS):
            for j in range(1, self.N_MBS+1):
                with self._sample_timer:
                    data = self._sample_data()

                data = {k: tf.convert_to_tensor(v) for k, v in data.items()}

                with self._train_timer:
                    terms = self.trainer.train(**data)

                kl = terms.pop('kl').numpy()
                value = terms.pop('value').numpy()

                for k, v in terms.items():
                    stats[f'train/{k}'].append(v.numpy())
                if getattr(self, '_max_kl', None) and kl > self._max_kl:
                    break

                self._after_train_step()

                if self._value_update == 'reuse':
                    self.dataset.update('value', value)

            if getattr(self, '_max_kl', None) and kl > self._max_kl:
                do_logging(f'Eearly stopping after {i*self.N_MBS+j} update(s) '
                    f'due to reaching max kl. Current kl={kl:.3g}', logger=logger)
                break

            if self._value_update == 'once':
                self.dataset.update_value_with_func(self.compute_value)
            if self._value_update is not None:
                last_value = self.compute_value()
                self.dataset.finish(last_value)

            self._after_train_epoch()
        n = i * self.N_MBS + j

        stats['misc/policy_updates'] = n
        stats['train/kl'] = kl
        stats['train/value'] = value,
        stats['time/sample_mean'] = self._sample_timer.average()
        stats['time/train_mean'] = self._train_timer.average()
        stats['time/fps'] = 1 / self._train_timer.average()
        
        if self._train_timer.total() > 1000:
            self._train_timer.reset()

        return n, stats

    def _train_extra_vf(self):
        stats = collections.defaultdict(list)
        for _ in range(self.N_VALUE_EPOCHS):
            for _ in range(self.N_MBS):
                data = self.dataset.sample(self._value_sample_keys)

                data = {k: tf.convert_to_tensor(data[k]) 
                    for k in self._value_sample_keys}

                terms = self.trainer.learn_value(**data)
                for k, v in terms.items():
                    stats[f'train/{k}'].append(v.numpy())
        return stats

    def _sample_data(self):
        return self.dataset.sample()

    def _after_train_step(self):
        """ Does something after each training step """
        pass

    def _after_train_epoch(self):
        """ Does something after each training epoch """
        pass

    def _store_additional_stats(self):
        stats = self.actor.get_auxiliary_stats()
        obs_rms, rew_rms = stats
        rms = {**obs_rms, 'reward': rew_rms}
        rms = {
            f'{k}_{kk}': vv for k, v in rms.items() \
                for kk, vv in v._asdict().items()
        }
        self.store(**rms)


class PPOStrategy(Strategy):
    def _post_init(self):
        self._value_input = None

    def record_inputs_to_vf(self, env_output):
        self._value_input = {
            'obs': self.actor.normalize_obs(env_output.obs['obs'])
        }

    def compute_value(self, value_inp: Dict[str, np.ndarray]=None):
        # be sure you normalize obs first if obs normalization is required
        if value_inp is None:
            value_inp = self._value_input
        value, _ = self.model.compute_value(**value_inp)
        return value.numpy()


create_strategy = functools.partial(
    create_strategy, 
    strategy_cls=PPOStrategy,
    training_loop_cls=PPOTrainingLoop
)
