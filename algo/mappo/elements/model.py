import os
import collections
import tensorflow as tf

from core.elements.model import Model, ModelEnsemble
from utility.file import source_file
from utility.tf_utils import assert_rank

source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class ModelImpl(Model):
    def encode(self, x, state, mask):
        if x.shape.ndims % 2 == 0:
            x = tf.expand_dims(x, 1)
        if mask.shape.ndims < 2:
            mask = tf.reshape(mask, (-1, 1))
        assert_rank(mask, 2)

        x = self.encoder(x)
        if hasattr(self, 'rnn'):
            x, state = self.rnn(x, state, mask)
        else:
            state = None
        if x.shape[1] == 1:
            x = tf.squeeze(x, 1)
        return x, state


class MAPPOActorModel(ModelImpl):
    @tf.function
    def action(self, obs, state, mask, action_mask=None,
            evaluation=False, return_eval_stats=False):
        assert obs.shape.ndims % 2 == 0, obs.shape

        x, state = self.encode(obs, state, mask)
        act_dist = self.policy(x, action_mask, evaluation=evaluation)
        action = self.policy.action(act_dist, evaluation)

        if evaluation:
            # we do not compute the value state at evaluation 
            return action, {}, state
        else:
            logpi = act_dist.log_prob(action)
            terms = {'logpi': logpi}
            return action, terms, state


class MAPPOValueModel(ModelImpl):
    @tf.function(experimental_relax_shapes=True)
    def compute_value(self, global_state, state, mask):
        x, state = self.encode(global_state, state, mask)
        value = self.value(x)
        return value, state


class MAPPOModelEnsemble(ModelEnsemble):
    def _post_init(self, config):
        state = {
            'mlstm': 'actor_h actor_c value_h value_c',
            'mgru': 'actor_h value_h',
        }
        self.state_type = collections.namedtuple(
            'State', state[self._rnn_type.split('_')[-1]])
        self.compute_value = self.value.compute_value

    @tf.function
    def action(self, actor_inp, value_inp, 
            evaluation=False, return_eval_stats=False):
        action, terms, actor_state = self.policy.action(**actor_inp, 
            evaluation=evaluation, return_eval_stats=return_eval_stats)
        value, value_state = self.value.compute_value(**value_inp)
        state = self.state_type(*actor_state, *value_state)
        if not evaluation:
            terms.update({
                'value': value,
            })
        return action, terms, state

    def split_state(self, state):
        mid = len(state) // 2
        actor_state, value_state = state[:mid], state[mid:]
        return self.policy.state_type(*actor_state), \
            self.value.state_type(*value_state)

    @property
    def state_size(self):
        return self.state_type(*self.policy.rnn.state_size, *self.value.rnn.state_size)

    @property
    def actor_state_size(self):
        return self.policy.rnn.state_size

    @property
    def value_state_size(self):
        return self.value.rnn.state_size

    @property
    def state_keys(self):
        return self.state_type(*self.state_type._fields)

    @property
    def actor_state_type(self):
        return self.policy.state_type
    
    @property
    def value_state_type(self):
        return self.value.state_type

    def reset_states(self, state=None):
        actor_state, value_state = self.split_state(state)
        self.policy.rnn.reset_states(actor_state)
        self.value.rnn.reset_states(value_state)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        actor_state = self.policy.rnn.get_initial_state(
            inputs, batch_size=batch_size, dtype=dtype)
        value_state = self.value.rnn.get_initial_state(
            inputs, batch_size=batch_size, dtype=dtype)
        return self.state_type(*actor_state, *value_state)


def create_model(config, env_stats, name='mappo'):
    config['policy']['policy']['action_dim'] = env_stats.action_dim
    config['policy']['policy']['is_action_discrete'] = env_stats.action_dim

    return MAPPOModelEnsemble(
        config=config, 
        name=name,
        policy=MAPPOActorModel,
        value=MAPPOValueModel
    )
