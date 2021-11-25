import os
import collections
import numpy as np
import tensorflow as tf

from core.elements.model import Model
from utility.file import source_file
from utility.tf_utils import assert_rank

# register ppo-related networks 
source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class PPOModel(Model):
    def _post_init(self):
        state = {
            'mlstm': 'action_h action_c h c',
            'mgru': 'action_h h',
        }
        self._state_type = collections.namedtuple(
            'State', state[self._rnn_type.split('_')[-1]])

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
               follow_mask, 
               bomb_mask,
               others_numbers=None, 
               others_jokers=None, 
               state=None, 
               mask=None,
               evaluation=False):
        x, state = self.encode(
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
        action_type, card_rank_mask, action_type_dist, card_rank_dist = \
            self.compute_policy_stream(
                x=x, 
                is_last_teammate_move=is_last_teammate_move,
                last_valid_action_type=last_valid_action_type, 
                rank=rank,
                action_type_mask=action_type_mask,
                follow_mask=follow_mask,
                bomb_mask=bomb_mask,
                evaluation=evaluation)
        card_rank = self.card_rank.action(card_rank_dist, evaluation)

        if evaluation:
            return (action_type, card_rank), {}, state
        else:
            action_type_logpi = action_type_dist.log_prob(action_type)
            card_rank_logpi = card_rank_dist.log_prob(card_rank)
            value = self.compute_value_stream(x, others_numbers, others_jokers)
            terms = {
                'card_rank_mask': card_rank_mask,
                'action_type_logpi': action_type_logpi, 
                'card_rank_logpi': card_rank_logpi, 
                'value': value
            }

            return (action_type, card_rank), terms, state    # keep the batch dimension for later use

    @tf.function
    def compute_value(self, 
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
                      others_numbers=None, 
                      others_jokers=None, 
                      state=None, 
                      mask=None):
        x, state = self.encode(
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
        value = self.compute_value_stream(x, others_numbers, others_jokers)
        
        return value, state

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
        last_action_numbers, last_action_jokers, \
            last_action_others, last_action_first_move, \
                last_action_filters = \
            [tf.reshape(x, (-1, *x.shape[2:])) for x in 
            [last_action_numbers, last_action_jokers, 
            last_action_others, last_action_first_move, last_action_filters]]
        assert_rank(last_action_numbers, 4)
        assert_rank([last_action_jokers, last_action_others], 3)
        
        x_a = self.action_encoder(
            last_action_numbers, last_action_jokers, last_action_others)
        action_mask = 1-last_action_first_move  # TODO: shall we do masking here?
        if state is None:
            action_state = None
        else:
            action_state, state = self.split_state(state)
        x_a, action_state = self.action_rnn(
            x_a, action_state, mask=action_mask, filter=last_action_filters)
        x_a = tf.reshape(x_a, (-1, numbers.shape[1], x_a.shape[-1]))

        others = tf.concat([left_cards, bombs_dealt], axis=-1)
        x_o = self.obs_encoder(numbers, jokers, others)
        assert_rank([x_o, x_a])
        x = tf.concat([x_o, x_a], axis=-1)
        x, state = self.rnn(x, state, mask)
        
        return x, self.state_type(*action_state, *state)

    def compute_policy_stream(self, 
                              x, 
                              is_last_teammate_move,
                              last_valid_action_type, 
                              rank, 
                              action_type_mask, 
                              follow_mask, 
                              bomb_mask, 
                              action_type=None, 
                              evaluation=False):
        action_type_aux = tf.concat(
            [last_valid_action_type, is_last_teammate_move], -1)
        action_type_dist = self.action_type(
            x, action_type_aux, action_type_mask, evaluation)
        if action_type is None:
            action_type = self.action_type.action(action_type_dist, evaluation)
        action_type_exp = tf.expand_dims(action_type, -1)
        assert_rank([action_type_exp, follow_mask, bomb_mask])
        card_rank_mask = tf.where(action_type_exp == 0, tf.ones_like(follow_mask),
            tf.where(action_type_exp == 1, follow_mask, bomb_mask))
        card_rank_dist = self.card_rank(x, rank, card_rank_mask)

        return action_type, card_rank_mask, action_type_dist, card_rank_dist

    def compute_value_stream(self, x, others_numbers, others_jokers):
        x_o = self.others_encoder(others_numbers, others_jokers)
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
            action_batch_size = np.prod(inputs['last_action_numbers'].shape[:2])
        else:
            action_batch_size = batch_size * sequential_dim
        action_state = self.action_rnn.get_initial_state(
            None, batch_size=action_batch_size, dtype=dtype)
        state = self.rnn.get_initial_state(
            inputs, batch_size=batch_size, dtype=dtype)
        state = self.state_type(*action_state, *state)
        return state

    @property
    def state_size(self):
        action_state = self.action_rnn.state_size
        state = self.rnn.state_size
        state = self.state_type(*action_state, *state)
        return state

    @property
    def state_keys(self):
        action_state = self.action_rnn.state_keys
        state = self.rnn.state_keys
        state = self.state_type(*action_state, *state)
        return state

    @property
    def state_type(self):
        return self._state_type


def create_model(config, env_stats, name='ppo', to_build=False):
    config.action_type.head.out_size = env_stats['action_dim']['action_type']
    config.card_rank.head.out_size = env_stats['action_dim']['card_rank']

    return PPOModel(
        config=config, 
        env_stats=env_stats, 
        name=name, 
        to_build=to_build)


if __name__ == '__main__':
    import os
    from tensorflow.keras import layers
    from env.func import create_env
    from utility.yaml_op import load_config
    config = load_config('algo/gd/configs/builtin.yaml')
    env = create_env(config['env'])
    env_stats = env.stats()
    model = create_model(config['model'], env_stats, name='ppo')
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