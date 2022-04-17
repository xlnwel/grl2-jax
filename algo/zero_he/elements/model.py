import os
import tensorflow as tf

from utility.file import source_file
from algo.hm.elements.model import ModelImpl, MAPPOActorModel, \
    MAPPOModelEnsemble as MAPPOModelBase

source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class MAPPOValueModel(ModelImpl):
    def compute_action_embedding(
        self, 
        global_state, 
        action
    ):
        raw_ae = self.action_embed(
            action, tile=True, mask_out_self=True, flatten=True)
        ae_weights = self.hyper_ae(global_state)
        eqt = 'bah,baho->bao' if len(raw_ae.shape) == 3 else 'btah,btaho->btao'
        ae = tf.einsum(eqt, raw_ae, ae_weights)

        return ae

    def compute_value_with_action_embeddings(
        self, 
        global_state, 
        ae,
        prev_reward, 
        prev_action, 
        state=None, 
        mask=None
    ):
        x = tf.concat([global_state, ae], -1)
        x, state = self.encode(
            x=x, 
            prev_reward=prev_reward, 
            prev_action=prev_action, 
            state=state, 
            mask=mask
        )
        value = self.value(x)

        return value, state

    def compute_value(
        self, 
        global_state, 
        action, 
        prev_reward, 
        prev_action, 
        state=None, 
        mask=None,
        return_ae=False
    ):
        ae = self.compute_action_embedding(global_state, action)
        value_a, state = self.compute_value_with_action_embeddings(
            global_state=global_state, 
            ae=ae,
            prev_reward=prev_reward, 
            prev_action=prev_action, 
            state=state, 
            mask=mask
        )

        # NOTE: we do not distinguish states of V(s) and V(s, a^{-i})
        # This is incorrect for RNNs, which we do not consider for now
        value, state = self.compute_value_with_action_embeddings(
            global_state=global_state, 
            ae=tf.zeros_like(ae),
            prev_reward=prev_reward, 
            prev_action=prev_action, 
            state=state, 
            mask=mask
        )

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
        action, terms, actor_state = self.policy.action(
            **actor_inp, state=actor_state,
            evaluation=evaluation, return_eval_stats=return_eval_stats)
        value, value_a, value_state = self.value.compute_value(
            **value_inp, action=action, state=value_state)
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
    config.value.hyper_ae.w_in = \
        config.value.action_embed.embed_size * env_stats.n_units

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
