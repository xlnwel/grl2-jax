import numpy as np
import collections

from core.elements.trainer import Trainer, TrainingLoopBase, create_trainer
from core.decorator import override
from core.tf_config import build
from algo.zero.elements.utils import get_data_format


class PPOTrainer(Trainer):
    @override(Trainer)
    def _build_train(self, env_stats):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = get_data_format(self.config, env_stats, self.loss.model, False)
        self.train = build(self.train, TensorSpecs)

    def raw_train(self, 
                  numbers, 
                  jokers, 
                  left_cards, 
                  is_last_teammate_move,
                  is_first_move,
                  last_valid_action_type,
                  rank,
                  bombs_dealt,
                  last_action_numbers, 
                  last_action_jokers, 
                  last_action_types,
                  last_action_rel_pids,
                  last_action_filters,
                  last_action_first_move,
                  action_type_mask, 
                  card_rank_mask, 
                  others_numbers, 
                  others_jokers, 
                  mask,
                  action_type, 
                  card_rank,
                  state,
                  value, 
                  traj_ret, 
                  advantage, 
                  action_type_logpi,
                  card_rank_logpi):
        tape, loss, terms = self.loss.loss(
            numbers=numbers, 
            jokers=jokers, 
            left_cards=left_cards, 
            is_last_teammate_move=is_last_teammate_move,
            is_first_move=is_first_move,
            last_valid_action_type=last_valid_action_type,
            rank=rank,
            bombs_dealt=bombs_dealt,
            last_action_numbers=last_action_numbers, 
            last_action_jokers=last_action_jokers, 
            last_action_types=last_action_types,
            last_action_rel_pids=last_action_rel_pids,
            last_action_filters=last_action_filters,
            last_action_first_move=last_action_first_move,
            action_type_mask=action_type_mask, 
            card_rank_mask=card_rank_mask, 
            others_numbers=others_numbers, 
            others_jokers=others_jokers, 
            action_type=action_type, 
            card_rank=card_rank,
            state=state, 
            mask=mask,
            value=value, 
            traj_ret=traj_ret, 
            advantage=advantage, 
            action_type_logpi=action_type_logpi,
            card_rank_logpi=card_rank_logpi)
        terms['norm'] = self.optimizer(tape, loss)

        return terms


class BCTrainer(Trainer):
    @override(Trainer)
    def _build_train(self, env_stats):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = get_data_format(self.config, env_stats, self.loss.model, False)
        self.train = build(self.train, TensorSpecs)

    @override(Trainer)
    def raw_train(self, 
                  numbers, 
                  jokers, 
                  left_cards, 
                  is_last_teammate_move,
                  is_first_move,
                  last_valid_action_type,
                  rank,
                  bombs_dealt,
                  last_action_numbers, 
                  last_action_jokers, 
                  last_action_types,
                  last_action_rel_pids,
                  last_action_filters,
                  last_action_first_move,
                  action_type_mask, 
                  card_rank_mask, 
                  others_numbers, 
                  others_jokers, 
                  action_type, 
                  card_rank,
                  state=None, 
                  mask=None):
        tape, loss, terms = self.loss.loss(
            numbers=numbers, 
            jokers=jokers, 
            left_cards=left_cards, 
            is_last_teammate_move=is_last_teammate_move,
            is_first_move=is_first_move,
            last_valid_action_type=last_valid_action_type,
            rank=rank,
            bombs_dealt=bombs_dealt,
            last_action_numbers=last_action_numbers, 
            last_action_jokers=last_action_jokers, 
            last_action_types=last_action_types,
            last_action_rel_pids=last_action_rel_pids,
            last_action_filters=last_action_filters,
            last_action_first_move=last_action_first_move,
            action_type_mask=action_type_mask, 
            card_rank_mask=card_rank_mask, 
            others_numbers=others_numbers, 
            others_jokers=others_jokers, 
            action_type=action_type, 
            card_rank=card_rank,
            state=state, 
            mask=mask)
        terms['norm'] = self.optimizer(tape, loss)

        return terms


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
                
                # for k, v in data.items():
                #     if k != 'card_rank_mask':
                #         stats[f'train/{k}'].append(v.numpy())

                with self._train_timer:
                    terms = self.trainer.train(**data)

                if self.config.training == 'ppo':
                    value = terms.pop('value').numpy()

                for k, v in terms.items():
                    stats[f'train/{k}'].append(v.numpy())

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


def create_trainer(config, loss, env_stats, *, name='ppo', **kwargs):
    if config['training'] == 'ppo' or config['training'] == 'pbt':
        trainer_cls = PPOTrainer
    elif config['training'] == 'bc':
        trainer_cls = BCTrainer
    else:
        raise ValueError(config['training'])
    trainer = trainer_cls(
        config=config, loss=loss, 
        env_stats=env_stats, name=name, **kwargs)

    return trainer


if __name__ == '__main__':
    import tensorflow as tf
    from tensorflow.keras import layers
    from env.func import create_env
    from utility.yaml_op import load_config
    from algo.gd.elements.model import create_model
    from algo.gd.elements.loss import create_loss

    config = load_config('algo/gd/configs/guandan.yaml')
    env = create_env(config['env'])
    env_stats = env.stats()
    model = create_model(config['model'], env_stats, name='ppo')
    loss = create_loss(config['loss'], model, name='ppo')
    trainer = create_trainer(config['trainer'], loss, env_stats, name='ppo')
    b = 2
    s = 3
    shapes = {
        k: (s, *v) for k, v in env_stats['obs_shape'].items()
    }
    dtypes = {
        k: v for k, v in env_stats['obs_dtype'].items()
    }
    x = {k: layers.Input(v, batch_size=b, dtype=dtypes[k]) 
        for k, v in shapes.items()}
    x['state'] = tuple(
        [layers.Input(config['model']['action_rnn']['units'], 
            batch_size=b*s, dtype=tf.float32) for _ in range(2)]
        + [layers.Input(config['model']['rnn']['units'], 
            batch_size=b, dtype=tf.float32) for _ in range(2)]
    )
    y = model.action(**x)
    model = tf.keras.Model(x, y)
    model.summary(200)
