import os
import collections
import tensorflow as tf

from core.elements.model import Model
from core.tf_config import build
from tools.file import source_file
from tools.tf_utils import assert_rank


# register networks 
source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class PPOModel(Model):
    def _pre_init(self, config, env_stats):
        self._oracle_value = config.get('oracle_value', False)
        if not self._oracle_value:
            config.pop('others_memory_encoder')

    def _build(self, env_stats, evaluation=False):
        basic_shape = (None,)
        shapes = env_stats['obs_shape']
        dtypes = env_stats['obs_dtype']
        TensorSpecs = {k: ((*basic_shape, *v), dtypes[k], k) 
            for k, v in shapes.items()}
        dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        TensorSpecs.update(dict(
            mask=(basic_shape, tf.float32, 'mask'),
            state=self.state_type(*[((None, sz), dtype, name) 
                for name, sz in self.state_size._asdict().items()]),
            evaluation=evaluation,
            return_eval_stats=evaluation,
        ))
        self.action = build(self.action, TensorSpecs)

    def _post_init(self):
        state = {
            'mlstm': 'action_h action_c h c',
            'mgru': 'action_h h',
        }
        self._state_type = collections.namedtuple(
            'State', state[self._rnn_type.split('_')[-1]])
        self._action_rnn_resets_every_round = self.config.get('action_rnn_resets_every_round', True)
    
    @tf.function
    def action(self, 
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
               mask=None,
               others_numbers=None, 
               others_jokers=None, 
               others_h=None,
               state=None, 
               evaluation=False,
               return_eval_stats=False):
        encoder_inp = self._add_seqential_dimension(
            numbers=numbers, 
            jokers=jokers, 
            left_cards=left_cards,
            bombs_dealt=bombs_dealt,
            last_action_numbers=last_action_numbers,
            last_action_jokers=last_action_jokers,
            last_action_types=last_action_types,
            last_action_rel_pids=last_action_rel_pids,
            last_action_filters=last_action_filters,
            last_action_first_move=last_action_first_move,
            mask=mask
        )
        x, state = self.encode(
            **encoder_inp,
            state=state)
        policy_inp = self._add_seqential_dimension(
            is_last_teammate_move=is_last_teammate_move,
            last_valid_action_type=last_valid_action_type, 
            rank=rank,
            action_type_mask=action_type_mask,
            card_rank_mask=card_rank_mask,
        )
        action_type, card_rank_mask, action_type_dist, card_rank_dist = \
            self.compute_policy_stream(
                x=x, 
                **policy_inp,
                evaluation=evaluation)
        card_rank = self.card_rank.action(card_rank_dist, evaluation)

        if evaluation:
            action_type, card_rank = tf.nest.map_structure(
                lambda x: tf.squeeze(x, 1), (action_type, card_rank)
            )
            return (action_type, card_rank), {}, state
        else:
            action_type_logpi = action_type_dist.log_prob(action_type)
            card_rank_logpi = card_rank_dist.log_prob(card_rank)
            value_inp = self._add_seqential_dimension(
                others_numbers=others_numbers,
                others_jokers=others_jokers,
                others_h=others_h,
            )
            value = self.compute_value_stream(x, **value_inp)
            terms = {
                # 'card_rank_mask': card_rank_mask,
                'action_type_logpi': action_type_logpi, 
                'card_rank_logpi': card_rank_logpi, 
                'value': value
            }
            action_type, card_rank = tf.nest.map_structure(
                lambda x: tf.squeeze(x, 1), (action_type, card_rank)
            )
            terms = tf.nest.map_structure(
                lambda x: tf.squeeze(x, 1), terms
            )
            return (action_type, card_rank), terms, state    # keep the batch dimension for later use

    def encode(self, 
               numbers, 
               jokers, 
               left_cards, 
               bombs_dealt,
               last_action_numbers, 
               last_action_jokers, 
               last_action_types,
               last_action_rel_pids,
               last_action_filters, 
               last_action_first_move, 
               state=None, 
               mask=None):
        assert_rank(numbers, 4)
        assert_rank([jokers, left_cards, bombs_dealt], 3)
        assert_rank(last_action_numbers, 5)
        assert_rank([last_action_jokers, last_action_types, last_action_rel_pids], 4)
        last_action_others = tf.concat(
            [last_action_types, last_action_rel_pids], axis=-1)
        if self._action_rnn_resets_every_round:
            last_action_numbers, last_action_jokers, \
                last_action_others, last_action_first_move, \
                    last_action_filters = \
                [tf.reshape(x, (-1, *x.shape[2:])) for x in 
                [last_action_numbers, last_action_jokers, 
                last_action_others, last_action_first_move, last_action_filters]]
            action_mask = 1 - last_action_first_move
        else:
            last_action_numbers, last_action_jokers, \
                last_action_others, action_mask, \
                    last_action_filters = \
                [tf.reshape(x, (-1, *x.shape[2:])) for x in 
                [last_action_numbers, last_action_jokers, 
                last_action_others, mask, last_action_filters]]
            action_mask = tf.tile(tf.expand_dims(action_mask, -1), [1, last_action_filters.shape[-1]])
        assert_rank(last_action_numbers, 4)
        assert_rank([last_action_jokers, last_action_others], 3)
        
        x_a = self.action_encoder(
            last_action_numbers, last_action_jokers, last_action_others)
        if state is None:
            action_state = None
        else:
            action_state, state = self.split_state(state)
        x_a, action_state = self.action_rnn(
            x_a, action_state, mask=action_mask, filter=last_action_filters)
        x_a = tf.reshape(x_a, (-1, numbers.shape[1], x_a.shape[-1]))

        others = tf.concat([left_cards, bombs_dealt], axis=-1)
        x_o = self.obs_encoder(numbers, jokers, others)
        assert_rank([x_o, x_a], 3)
        x = tf.concat([x_o, x_a], axis=-1)
        x, state = self.rnn(x, state, mask)
        
        return x, self.state_type(*action_state, *state)

    def compute_policy_stream(self, 
                              x, 
                              is_last_teammate_move,
                              last_valid_action_type, 
                              rank, 
                              action_type_mask, 
                              card_rank_mask, 
                              action_type=None, 
                              evaluation=False):
        action_type_aux = tf.concat(
            [last_valid_action_type, is_last_teammate_move, rank], -1)
        action_type_dist = self.action_type(
            x, action_type_aux, action_type_mask, evaluation)
        if action_type is None:
            action_type = self.action_type.action(action_type_dist, evaluation)
        if len(card_rank_mask.shape) > len(rank.shape):
            card_rank_mask = tf.gather(card_rank_mask, action_type, axis=2, batch_dims=2)
        assert_rank([x, rank, card_rank_mask])
        card_rank_dist = self.card_rank(x, rank, card_rank_mask)

        return action_type, card_rank_mask, action_type_dist, card_rank_dist

    def compute_value_stream(self, 
                            x, 
                            others_numbers, 
                            others_jokers,
                            others_h):
        x_o = self.others_encoder(others_numbers, others_jokers)
        if self._oracle_value:
            x_h = self.others_memory_encoder(others_h)
            x_v = tf.concat([x, x_o, x_h], axis=-1)
        else:
            x_v = tf.concat([x, x_o], axis=-1)
        value = self.value(x_v)
        return value

    def split_state(self, state):
        mid = len(state) // 2
        action_state, value_state = state[:mid], state[mid:]
        
        return self.action_rnn.state_type(*action_state), \
            self.rnn.state_type(*value_state)

    def get_initial_state(self, inputs=None, batch_size=None, sequential_dim=None, dtype=None):
        if inputs is not None:
            batch_size = inputs['last_action_numbers'].shape[0]
            sequential_dim = inputs['last_action_numbers'].shape[1]
        action_batch_size = batch_size * sequential_dim
        action_state = self.action_rnn.get_initial_state(
            batch_size=action_batch_size, dtype=dtype)
        state = self.rnn.get_initial_state(
            batch_size=batch_size, dtype=dtype)
        state = self.state_type(*action_state, *state)
        return state

    @property
    def state_size(self):
        action_state_size = self.action_rnn.state_size
        state_size = self.rnn.state_size
        state_size = self.state_type(*action_state_size, *state_size)
        return state_size

    @property
    def state_keys(self):
        return self.state_type(*self._state_type._fields)

    @property
    def state_type(self):
        return self._state_type

    def _add_seqential_dimension(self, add_sequential_dim=True, **kwargs):
        if add_sequential_dim:
            return tf.nest.map_structure(lambda x: tf.expand_dims(x, 1) 
                if isinstance(x, tf.Tensor) else x, kwargs)
        else:
            return kwargs


class PPGMixin:
    def _build(self, env_stats, evaluation=False):
        basic_shape = (None,)
        shapes = env_stats['obs_shape']
        dtypes = env_stats['obs_dtype']
        TensorSpecs = {k: ((*basic_shape, *v), dtypes[k], k) 
            for k, v in shapes.items()}
        dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        TensorSpecs.update(dict(
            state=self.state_type(*[((None, sz), dtype, name) 
                for name, sz in self.state_size._asdict().items()]),
            evaluation=evaluation,
            return_eval_stats=evaluation,
        ))
        self.action = build(self.action, TensorSpecs)

        basic_shape = (None, self.config.sample_size)
        TensorSpecs = {k: ((*basic_shape, *v), dtypes[k], k) 
            for k, v in shapes.items()}
        TensorSpecs['action_type'] = (basic_shape, tf.int32, 'action_type')
        TensorSpecs.update({
            name: ((None, sz), tf.float32, name)
                for name, sz in self.state_size._asdict().items()
        })
        self.compute_logits_values = build(self.compute_logits_values, TensorSpecs)

    @tf.function
    def compute_logits_values(self,
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
                            mask,
                            others_numbers, 
                            others_jokers, 
                            others_h, 
                            action_type,
                            action_h,
                            action_c,
                            h,
                            c):
        state = (action_h, action_c, h, c)
        x, _ = self.encode(
            numbers=numbers, 
            jokers=jokers, 
            left_cards=left_cards,
            bombs_dealt=bombs_dealt,
            last_action_numbers=last_action_numbers,
            last_action_jokers=last_action_jokers,
            last_action_types=last_action_types,
            last_action_rel_pids=last_action_rel_pids,
            last_action_filters=last_action_filters,
            last_action_first_move=last_action_first_move,
            state=state,
            mask=mask)
        _, card_rank_mask, action_type_dist, card_rank_dist = \
            self.compute_policy_stream(
                x=x, 
                is_last_teammate_move=is_last_teammate_move,
                last_valid_action_type=last_valid_action_type, 
                rank=rank,
                action_type_mask=action_type_mask,
                card_rank_mask=card_rank_mask,
                action_type=action_type,
                evaluation=False)
        value = self.compute_value_stream(
            x,
            others_numbers=others_numbers,
            others_jokers=others_jokers,
            others_h=others_h
        )
        return action_type_dist.logits, card_rank_dist.logits, value


class PPGModel(PPGMixin, PPOModel):
    pass


def create_model(
        config, 
        env_stats, 
        name='gd_zero', 
        to_build=False, 
        to_build_for_eval=False, 
        PPOModel=PPOModel, 
        PPGModel=PPGModel):
    config.action_type.head.out_size = env_stats['action_dim']['action_type']
    config.card_rank.head.out_size = env_stats['action_dim']['card_rank']

    training = config.get('training', 'ppo')
    if training == 'ppo':
        model = PPOModel
    elif training == 'ppg' or training == 'pbt':
        model = PPGModel
    else:
        raise ValueError(training)
    return model(
        config=config, 
        env_stats=env_stats, 
        name=name, 
        to_build=to_build,
        to_build_for_eval=to_build_for_eval)


if __name__ == '__main__':
    import os
    from tensorflow.keras import layers
    from env.func import create_env
    from tools.yaml_op import load_config
    config = load_config('algo/gd/configs/builtin.yaml')
    env = create_env(config['env'])
    env_stats = env.stats()
    model = create_model(config['model'], env_stats, name='gd_zero')
    b = 2
    s = 3
    shapes = {
        k: (s, *v) for k, v in env_stats['obs_shape'].items()
        if k != 'is_last_teammate_move' and k != 'is_first_move'
    }
    dtypes = {
        k: v for k, v in env_stats['obs_dtype'].items()
        if k != 'is_last_teammate_move' and k != 'is_first_move'
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
