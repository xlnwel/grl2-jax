import copy
import collections
import functools
import logging
from typing import Dict
import numpy as np
import tensorflow as tf

from core.elements.strategy import Strategy, create_strategy
from core.log import do_logging
from core.mixin.strategy import TrainingLoopBase, Memory


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
                
                for k, v in data.items():
                    if k != 'card_rank_mask':
                        stats[f'train/{k}'].append(v.numpy())

                with self._train_timer:
                    terms = self.trainer.train(**data)

                if self.config.training == 'ppo':
                    value = terms.pop('value').numpy()

                for k, v in terms.items():
                    stats[f'train/{k}'].append(v.numpy())
                
                action_type = data['action_type'].numpy()
                original_card_rank_mask = data['card_rank_mask']
                card_rank_mask = stats['train/card_rank_mask'][-1]
                for b in range(action_type.shape[0]):
                    np.testing.assert_equal(original_card_rank_mask[b, 0, action_type[b, 0]], card_rank_mask[b, 0])
                
                if self.config.training == 'ppo':
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

        stats['misc/policy_updates'] = n
        # stats['train/kl'] = kl
        # stats['train/value'] = value,
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


class PPOStrategy(Strategy):
    def _post_init(self):
        self._value_input = None

        self._memories = {}

    """ Calling Methods """
    def _prepare_input_to_actor(self, env_output):
        """ Extract data from env_output as the input 
        to Actor for inference """
        inp = self._add_memory_state_to_input(env_output)
        return inp

    def _record_output(self, out):
        states = out[-1]
        for i, memory in enumerate(self._last_memories):
            state = self.model.state_type(*[s[i:i+1] for s in states])
            memory.reset_states(state)

    """ PPO Methods """
    def record_inputs_to_vf(self, env_output):
        env_output = copy.deepcopy(env_output)
        self._value_input = self._add_memory_state_to_input(env_output)

    def compute_value(self, value_inp: Dict[str, np.ndarray]=None):
        # be sure you normalize obs first if obs normalization is required
        if value_inp is None:
            value_inp = self._value_input
        value, _ = self.model.compute_value(**value_inp)
        return value.numpy()

    def _add_memory_state_to_input(self, env_output):
        inp = env_output.obs
        eids = inp.pop('eid')
        pids = inp.pop('pid')

        states = []
        self._last_memories = []
        for eid, pid in zip(eids, pids):
            if eid not in self._memories:
                self._memories[eid] = [Memory(self.model) for _ in range(4)]
            self._last_memories.append(self._memories[eid][pid])
            states.append(self._memories[eid][pid].get_states_for_inputs(
                batch_size=1, sequential_dim=1
            ))
        state = self.model.state_type(*[np.concatenate(s) for s in zip(*states)])
        
        # TODO: remove the following as we're gonna reset states in LSTM anyway
        # state = self._memories[0][0].apply_mask_to_state(state, mask)
        inp['state'] = state
        
        return inp


create_strategy = functools.partial(
    create_strategy, 
    strategy_cls=PPOStrategy,
    training_loop_cls=PPOTrainingLoop
)
