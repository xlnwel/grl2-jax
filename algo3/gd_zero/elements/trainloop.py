import collections
import tensorflow as tf

from core.elements.trainloop import TrainingLoopBase


class PPOTrainingLoop(TrainingLoopBase):
    def _post_init(self):
        pass

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

                action_type_kl = terms.pop('action_type_kl').numpy()
                card_rank_kl = terms.pop('card_rank_kl').numpy()
                value = terms.pop('value').numpy()

                for k, v in terms.items():
                    stats[f'train/{k}'].append(v.numpy())
                stats['train/value'].append(value.mean())

                self._after_train_step()

                if self._value_update == 'reuse':
                    self.dataset.update('value', value)

            if self._value_update == 'once':
                self.dataset.update_value_with_func(self.compute_value)
            if self._value_update is not None:
                last_value = self.compute_value()
                self.dataset.finish(last_value)

            self._after_train_epoch()
        n = i * self.N_MBS + j

        stats.update({
            'train/action_type_kl': action_type_kl,
            'train/card_rank_kl': card_rank_kl})
        stats['time/sample_mean'] = self._sample_timer.average()
        stats['time/train_mean'] = self._train_timer.average()
        stats['time/tps'] = 1 / self._train_timer.last()
        
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


class PPGTrainingLoop(PPOTrainingLoop):
    def _post_init(self):
        self.N_AUX_MBS = self.N_SEGS * self.N_AUX_MBS_PER_SEG

    def aux_train_record(self):
        stats = collections.defaultdict(list)

        for i in range(self.N_AUX_EPOCHS):
            for j in range(1, self.N_AUX_MBS+1):
                data = self.dataset.sample()
                data = {k: tf.convert_to_tensor(v) for k, v in data.items()}
                terms = self.trainer.aux_train(**data)

                action_type_kl = terms.pop('action_type_kl').numpy()
                card_rank_kl = terms.pop('card_rank_kl').numpy()
                value = terms.pop('value').numpy()

                for k, v in terms.items():
                    stats[f'aux/{k}'].append(v.numpy())
                stats['aux/value'].append(value.mean())

                if self._value_update == 'reuse':
                    self.dataset.aux_update('value', value)
            if self._value_update == 'once':
                self.dataset.compute_aux_data_with_func(self.compute_value)
            if self._value_update is not None:
                last_value = self.compute_value()
                self.dataset.aux_finish(last_value)

        stats.update({
            'aux/action_type_kl': action_type_kl,
            'aux/card_rank_kl': card_rank_kl})
        return stats

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