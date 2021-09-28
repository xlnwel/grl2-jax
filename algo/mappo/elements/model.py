import os
import collections
import numpy as np
import tensorflow as tf

from core.module import Model, ModelEnsemble
from env.typing import EnvOutput
from utility.file import source_file
from utility.tf_utils import assert_rank, tensor2numpy

source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class ModelImpl(Model):
    def encode(self, x, state, mask, prev_action=None, prev_reward=None):
        if x.shape.ndims % 2 == 0:
            x = tf.expand_dims(x, 1)
        if mask.shape.ndims < 2:
            mask = tf.reshape(mask, (-1, 1))
        assert_rank(mask, 2)

        x = self.encoder(x)
        additional_rnn_input = self._process_additional_input(
            x, prev_action, prev_reward)
        x, state = self.rnn(x, state, mask, 
            additional_input=additional_rnn_input)
        if x.shape[1] == 1:
            x = tf.squeeze(x, 1)
        return x, state

    def _process_additional_input(self, x, prev_action, prev_reward):
        """ NOTE: This method is not tested. """
        results = []
        if prev_action is not None:
            if self.actor.is_action_discrete:
                if prev_action.shape.ndims < 2:
                    prev_action = tf.reshape(prev_action, (-1, 1))
                prev_action = tf.one_hot(
                    prev_action, self.actor.action_dim, dtype=x.dtype)
            else:
                if prev_action.shape.ndims < 3:
                    prev_action = tf.reshape(
                        prev_action, (-1, 1, self.actor.action_dim))
            assert_rank(prev_action, 3)
            results.append(prev_action)
        if prev_reward is not None:
            if prev_reward.shape.ndims < 3:
                prev_reward = tf.reshape(prev_reward, (-1, 1, 1))
            assert_rank(prev_reward, 3)
            results.append(prev_reward)
        assert_rank(results, 3)
        return results


class MAPPOActorModel(ModelImpl):
    @tf.function
    def action(self, obs, state, mask, action_mask=None,
            prev_action=None, prev_reward=None, 
            evaluation=False, return_eval_stats=False):
        assert obs.shape.ndims % 2 == 0, obs.shape

        x, state = self.encode(
            obs, state, mask, prev_action, prev_reward)
        act_dist = self.actor(x, action_mask, evaluation=evaluation)
        action = self.actor.action(act_dist, evaluation)

        if evaluation:
            # we do not compute the value state at evaluation 
            return action, state
        else:
            logpi = act_dist.log_prob(action)
            terms = {'logpi': logpi}
            out = (action, terms)
            return out, state


class MAPPOValueModel(ModelImpl):
    @tf.function(experimental_relax_shapes=True)
    def compute_value(self, global_state, state, mask, 
            prev_action=None, prev_reward=None):
        x, state = self.encode(
            global_state, state, mask, prev_action, prev_reward)
        value = self.value(x)
        return value, state


class MAPPOModelEnsemble(ModelEnsemble):
    def _post_init(self, config):
        self._setup_rms_stats()
        self._setup_memory_state_record()
        state = {
            'mlstm': 'actor_h actor_c value_h value_c',
            'mgru': 'actor_h value_h',
        }
        self.state_type = collections.namedtuple(
            'State', state[self._rnn_type.split('_')[-1]])

    def compute_value(self, global_state, state, mask, return_state):
        value, state = self.value.compute_value(
            global_state=global_state, 
            state=state,
            mask=mask,
        )
        return tensor2numpy((value, state)) \
            if return_state else value.numpy()

    def reshape_env_output(self, env_output):
        """ merges the batch and agent dimensions """
        obs, reward, discount, reset = env_output
        new_obs = {}
        for k, v in obs.items():
            new_obs[k] = np.concatenate(v)
        # reward and discount are not used for inference so we do not process them
        # reward = np.concatenate(reward)
        # discount = np.concatenate(discount)
        reset = np.concatenate(reset)
        return EnvOutput(new_obs, reward, discount, reset)

    def _process_input(self, inp: dict, evaluation: bool):
        def divide_obs(obs):
            actor_inp = obs.copy()
            actor_inp.pop('global_state')
            actor_inp.pop('life_mask')
            value_inp = {
                'global_state': obs['global_state']
            }
            return actor_inp, value_inp

        env_output = self.reshape_env_output(env_output)
        if evaluation:
            obs = self.process_obs_with_rms(env_output.obs, update_rms=False)
        else:
            life_mask = env_output.obs.get('life_mask')
            obs = self.process_obs_with_rms(env_output.obs, mask=life_mask)
        mask = self._get_mask(env_output.reset)
        actor_inp, value_inp = divide_obs(obs)
        state = self._get_state_with_batch_size(mask.shape[0])
        actor_state, value_state = self.split_state(state)
        actor_inp = self._add_memory_state_to_input(
            actor_inp, mask=mask, state=actor_state)
        value_inp = self._add_memory_state_to_input(
            value_inp, mask=mask, state=value_state)
        return {'actor': actor_inp, 'value': value_inp}

    @tf.function
    def action(self, inp, **kwargs):
        print('retracing MAPPOModelEnsemble action')
        actor_inp, value_inp = inp
        out, actor_state = self.actor.action(**actor_inp, **kwargs)
        value, value_state = self.value.compute_value(**value_inp)
        out[1]['value'] = value
        state = self.state_type(*actor_state, *value_state)
        return out, state

    def _process_output(self, inp, out, evaluation):
        out = self._add_tensors_to_terms(inp, out, evaluation)
        out = tensor2numpy(out)
        out = self._add_non_tensors_to_terms(inp, out, evaluation)
        return out

    def _add_non_tensors_to_terms(self, inp, out, evaluation):
        print('add non tensors to terms', list(inp), list(out))
        if not evaluation:
            out[1].update({
                'obs': inp[0]['obs'], # ensure obs is placed in terms even when no observation normalization is performed
                'global_state': inp[1]['global_state'],
                'mask': inp[0]['mask'],
            })
            if 'action_mask' in inp[0]:
                out[1]['action_mask'] = inp[0]['action_mask']
            if 'life_mask' in inp[0]:
                out[1]['life_mask'] = inp[0]['life_mask']
        return out

    def split_state(self, state):
        mid = len(state) // 2
        actor_state, value_state = state[:mid], state[mid:]
        return self.actor.state_type(*actor_state), \
            self.value.state_type(*value_state)

    @property
    def state_size(self):
        return self.state_type(*self.actor.rnn.state_size, *self.value.rnn.state_size)

    @property
    def actor_state_size(self):
        return self.actor.rnn.state_size

    @property
    def value_state_size(self):
        return self.value.rnn.state_size

    @property
    def state_keys(self):
        return self.state_type(*self.state_type._fields)

    def reset_states(self, state=None):
        actor_state, value_state = self.split_state(state)
        self.actor.rnn.reset_states(actor_state)
        self.value.rnn.reset_states(value_state)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        actor_state = self.actor.rnn.get_initial_state(
            inputs, batch_size=batch_size, dtype=dtype)
        value_state = self.value.rnn.get_initial_state(
            inputs, batch_size=batch_size, dtype=dtype)
        return self.state_type(*actor_state, *value_state)


def create_model(config, env_stats, name='mappo'):
    config['actor']['actor']['action_dim'] = env_stats.action_dim
    config['actor']['actor']['is_action_discrete'] = env_stats.action_dim

    return MAPPOModelEnsemble(
        config=config, 
        name=name,
        actor=MAPPOActorModel,
        value=MAPPOValueModel
    )
