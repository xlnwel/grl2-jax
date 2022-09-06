import os
import tensorflow as tf

from tools.file import source_file
from algo.gpo.elements.model import ModelImpl, PPOActorModel, \
    PPOModelEnsemble as PPOModelBase

source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class PPOValueModel(ModelImpl):
    def compute_action_embedding(
        self, 
        global_state, 
        action,
        life_mask,
        multiply=False
    ):
        raw_ae = self.action_embed(
            action, 
            multiply=multiply, 
            tile=True, 
            mask_out_self=True, 
            flatten=True, 
            mask=life_mask
        )
        ae = self.ae_encoder(raw_ae)

        return ae

    def compute_value_with_action_embeddings(
        self, 
        global_state, 
        ae,
        prev_reward=None, 
        prev_action=None, 
        state=None, 
        mask=None
    ):
        x = global_state if self.config.concat_end \
            else tf.concat([global_state, ae], -1)
        x, state = self.encode(
            x=x, 
            prev_reward=prev_reward, 
            prev_action=prev_action, 
            state=state, 
            mask=mask
        )

        value = self.value(x)

        return value, state

    @tf.function
    def compute_value(
        self, 
        global_state, 
        action, 
        pi, 
        prev_reward=None, 
        prev_action=None, 
        life_mask=None, 
        state=None, 
        mask=None, 
        return_ae=False
    ):
        if hasattr(self, 'rnn'):
            T, U, F = global_state.shape[1:]
            gs_shaped = tf.reshape(global_state, (-1, U, F))
            a_shaped = tf.reshape(action, (-1, *action.shape[2:]))
            lm_shaped = tf.reshape(life_mask, (-1, U))
        else:
            gs_shaped = global_state
            a_shaped = action
            lm_shaped = life_mask
        ae = self.compute_action_embedding(
            gs_shaped, 
            a_shaped, 
            lm_shaped
        )

        if self.config.v_pi:
            if hasattr(self, 'rnn'):
                pi_shaped = tf.reshape(pi, (-1, *pi.shape[2:]))
            else:
                pi_shaped = pi
            pi_ae = self.compute_action_embedding(
                gs_shaped, 
                pi_shaped, 
                lm_shaped, 
                True
            )
        else:
            pi_ae = tf.zeros_like(ae)
        
        if hasattr(self, 'rnn'):
            ae = tf.reshape(ae, (-1, T, *ae.shape[1:]))
            pi_ae = tf.reshape(pi_ae, (-1, T, *pi_ae.shape[1:]))

        if self.config.concat_end:
            x = global_state
            x, state = self.encode(
                x=x, 
                prev_reward=prev_reward, 
                prev_action=prev_action, 
                state=state, 
                mask=mask
            )
            value_a = self.value(tf.concat([x, ae], -1))
            value = self.value(tf.concat([x, pi_ae], -1))
        else:
            assert not hasattr(self, 'rnn'), 'rnn state is incorrect'
            x = tf.concat([global_state, ae], -1)
            x, state = self.encode(
                x=x, 
                prev_reward=prev_reward, 
                prev_action=prev_action, 
                state=state, 
                mask=mask
            )
            value_a = self.value(x)

            x = tf.concat([global_state, pi_ae], -1)
            x, state = self.encode(
                x=x, 
                prev_reward=prev_reward, 
                prev_action=prev_action, 
                state=state, 
                mask=mask
            )
            value = self.value(x)

        if return_ae:
            return ae, value, value_a, state
        else:
            return value, value_a, state


class PPOModelEnsemble(PPOModelBase):
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
            evaluation=evaluation, 
            return_eval_stats=return_eval_stats)
        value, value_a, value_state = self.value.compute_value(
            **value_inp, action=action, pi=terms.get('pi'), state=value_state)
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
        name='ppo', 
        to_build=False,
        to_build_for_eval=False,
        **kwargs):
    if 'aid' in config:
        aid = config['aid']
        config.policy.policy.action_dim = env_stats.action_dim[aid]
        config.policy.policy.is_action_discrete = env_stats.is_action_discrete[aid]
        config.value.action_embed.input_dim = env_stats.action_dim[aid]
    else:
        config.policy.policy.action_dim = env_stats.action_dim
        config.policy.policy.is_action_discrete = env_stats.is_action_discrete
        config.value.action_embed.input_dim = env_stats.action_dim
    config.value.action_embed.input_length = env_stats.n_units

    if config['actor_rnn_type'] is None:
        config['policy'].pop('rnn', None)
    else:
        config['policy']['rnn']['nn_id'] = config['actor_rnn_type']
    if config['value_rnn_type'] is None:
        config['value'].pop('rnn', None)
    else:
        config['value']['rnn']['nn_id'] = config['value_rnn_type']

    return PPOModelEnsemble(
        config=config, 
        env_stats=env_stats, 
        name=name,
        to_build=to_build, 
        to_build_for_eval=to_build_for_eval,
        policy=PPOActorModel,
        value=PPOValueModel,
        **kwargs
    )
