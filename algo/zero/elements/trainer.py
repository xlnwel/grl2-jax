import tensorflow as tf

from core.elements.trainer import Trainer, create_trainer
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


class PPGTrainer(Trainer):
    @override(Trainer)
    def _build_train(self, env_stats):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = get_data_format(self.config, env_stats, self.loss.model, True)
        self.train = build(self.train, TensorSpecs)
        TensorSpecs.pop('card_rank')
        TensorSpecs.pop('advantage')
        TensorSpecs.pop('action_type_logpi')
        TensorSpecs.pop('card_rank_logpi')
        basic_shape = (None, self.config['sample_size'])
        TensorSpecs['action_type_logits'] = ((*basic_shape, env_stats['action_dim']['action_type']), tf.float32, 'action_type_logits')
        TensorSpecs['card_rank_logits']=((*basic_shape, env_stats['action_dim']['card_rank']), tf.float32, 'card_rank_logits')
        self.aux_train = build(self.raw_aux_train, TensorSpecs)

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
                  value, 
                  traj_ret, 
                  advantage, 
                  action_type_logpi,
                  card_rank_logpi,
                  action_h,
                  action_c,
                  h,
                  c):
        state = (action_h, action_c, h, c)
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

    @tf.function
    def raw_aux_train(self,
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
                    value, 
                    traj_ret, 
                    action_type_logits,
                    card_rank_logits,
                    action_h,
                    action_c,
                    h,
                    c):
        state = (action_h, action_c, h, c)
        tape, loss, terms = self.loss.aux_loss(
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
            state=state, 
            mask=mask,
            value=value, 
            traj_ret=traj_ret, 
            action_type_logits=action_type_logits,
            card_rank_logits=card_rank_logits)
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


def create_trainer(config, loss, env_stats, *, name='ppo', **kwargs):
    if config['training'] == 'ppg' or config['training'] == 'pbt':
        trainer_cls = PPGTrainer
    elif config['training'] == 'ppo':
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
