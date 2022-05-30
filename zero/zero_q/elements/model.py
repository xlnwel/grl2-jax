import os
import tensorflow as tf

from algo.zero.elements.model import ModelImpl, MAPPOActorModel, \
    MAPPOModelEnsemble as MAPPOModelBase
from utility.file import source_file
from utility.tf_utils import assert_rank_and_shape_compatibility

source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class MAPPOValueModel(ModelImpl):
    def compute_action_embedding(
        self, 
        action,
        life_mask,
    ):
        raw_ae = self.action_embed(
            action, 
            tile=True, 
            mask_out_self=True, 
            flatten=True, 
            mask=life_mask
        )
        ae = self.ae_encoder(raw_ae)

        return ae

    def compute_expected_value(self, x, probs):
        q = self.value(x)
        assert_rank_and_shape_compatibility([q, probs])
        value = tf.reduce_mean(q * probs, -1)
        return value

    def compute_value(
        self, 
        global_state, 
        action, 
        pi, 
        prev_reward, 
        prev_action,
        life_mask=None, 
        state=None, 
        mask=None, 
        return_ae=False,
        return_v=True
    ):
        ae = self.compute_action_embedding(action, life_mask)
        x_a = tf.concat([global_state, ae], -1)
        x_a, state = self.encode(
            x=x_a, 
            prev_reward=prev_reward, 
            prev_action=prev_action, 
            state=state, 
            mask=mask
        )
        value_a = self.compute_expected_value(x_a, pi) \
            if return_v else self.value(x_a)

        if self.config.v_pi:
            pi_ae = self.compute_action_embedding(pi, life_mask)
            x = tf.concat([global_state, pi_ae], -1)
        else:
            x = tf.concat([global_state, tf.zeros_like(ae)], -1)
        x, state = self.encode(
            x=x, 
            prev_reward=prev_reward, 
            prev_action=prev_action, 
            state=state, 
            mask=mask
        )
        value = self.compute_expected_value(x, pi) \
            if return_v else self.value(x)

        # x, state = self.encode(
        #     x=global_state, 
        #     prev_reward=prev_reward, 
        #     prev_action=prev_action, 
        #     state=state, 
        #     mask=mask
        # )
        # ae = self.action_embed(action)
        # x_a = tf.concat([x, ae], -1)
        # value_a = self.value(x_a)
        # x = tf.concat([x, tf.zeros_like(ae)], -1)
        # value = self.value(x)

        if return_ae:
            return ae, value, value_a, state
        else:
            return value, value_a, state


class MAPPOModelEnsemble(MAPPOModelBase):
    @tf.function
    def action(
        self, 
        actor_inp: dict, 
        value_inp: dict, 
        actor_state: tuple=None, 
        value_state: tuple=None,
        evaluation: bool=False, 
        return_eval_stats: bool=False
    ):
        if self.actor_has_rnn:
            actor_inp, = self._add_seqential_dimension(actor_inp)
        if self.value_has_rnn:
            value_inp, = self._add_seqential_dimension(value_inp)
        (action, pi), terms, actor_state = self.policy.action(
            **actor_inp, state=actor_state,
            evaluation=evaluation, 
            return_eval_stats=return_eval_stats)
        value, value_a, value_state = self.value.compute_value(
            **value_inp, action=action, pi=pi, state=value_state)
        state = self.state_type(actor_state, value_state) \
            if self.has_rnn else None
        if self.actor_has_rnn:
            action, terms = self._remove_seqential_dimension(action, terms)
        if self.value_has_rnn:
            value, value_a = self._remove_seqential_dimension(value, value_a)
        if not evaluation:
            terms.update({
                'value': value,
                'value_a': value_a,
            })
        return action, terms, state


def create_model(
        config, 
        env_stats, 
        name='mappo', 
        to_build=False,
        to_build_for_eval=False,
        **kwargs):
    aid = config['aid']
    config.policy.policy.action_dim = env_stats.action_dim[aid]
    config.policy.policy.is_action_discrete = env_stats.is_action_discrete[aid]
    config.value.action_embed.input_dim = env_stats.action_dim[aid]
    config.value.action_embed.input_length = env_stats.n_units
    config.value.value.out_size = env_stats.action_dim[aid]

    if config['actor_rnn_type'] is None:
        config['policy'].pop('rnn', None)
    else:
        config['policy']['rnn']['nn_id'] = config['actor_rnn_type']
    if config['value_rnn_type'] is None:
        config['value'].pop('rnn', None)
    else:
        config['value']['rnn']['nn_id'] = config['value_rnn_type']

    return MAPPOModelEnsemble(
        config=config, 
        env_stats=env_stats, 
        name=name,
        to_build=to_build, 
        to_build_for_eval=to_build_for_eval,
        policy=MAPPOActorModel,
        value=MAPPOValueModel,
        **kwargs
    )
