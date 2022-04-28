import os
import tensorflow as tf

from core.tf_config import build
from utility.file import source_file
from algo.hm.elements.model import ModelImpl, PPOActorModel, \
    PPOModelEnsemble as PPOModelBase

source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class PPOActorModel(ModelImpl):
    def action(
        self, 
        obs, 
        prev_reward=None, 
        prev_action=None, 
        state=None, 
        mask=None, 
        action_mask=None,
        evaluation=False, 
        return_eval_stats=False
    ):
        x, state = self.encode(
            x=obs, 
            prev_reward=prev_reward, 
            prev_action=prev_action, 
            state=state, 
            mask=mask
        )
        act_dist = self.policy(x, action_mask, evaluation=evaluation)
        action = self.policy.action(act_dist, evaluation)
        pi = tf.nn.softmax(act_dist.logits)
        if evaluation:
            # we do not compute the value state at evaluation 
            return (action, pi), {}, state
        else:
            logprob = act_dist.log_prob(action)
            terms = {
                'logprob': logprob, 
            }
            return (action, pi), terms, state


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
        pi, 
        prev_reward, 
        prev_action, 
        life_mask=None, 
        state=None, 
        mask=None, 
        return_ae=False
    ):
        ae = self.compute_action_embedding(
            global_state, 
            action, 
            life_mask
        )
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
        if self.config.v_pi:
            pi_ae = self.compute_action_embedding(
                global_state, 
                pi, 
                life_mask, 
                True
            )
        else:
            pi_ae = tf.zeros_like(ae)
        value, state = self.compute_value_with_action_embeddings(
            global_state=global_state, 
            ae=pi_ae,
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


class PPOModelEnsemble(PPOModelBase):
    def _build(self, env_stats, evaluation=False):
        aid = self.config.aid
        basic_shape = (None, len(env_stats.aid2uids[aid]))
        dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        actor_inp=dict(
            obs=((*basic_shape, *env_stats.obs_shape[aid]['obs']), 
                env_stats.obs_dtype[aid]['obs'], 'obs'),
            prev_reward=(basic_shape, tf.float32, 'prev_reward'), 
            prev_action=((*basic_shape, env_stats.action_dim[aid]), tf.float32, 'prev_action'), 
        )
        value_inp=dict(
            global_state=(
                (*basic_shape, *env_stats.obs_shape[aid]['global_state']), 
                env_stats.obs_dtype[aid]['global_state'], 'global_state'), 
            prev_reward=(basic_shape, tf.float32, 'prev_reward'), 
            prev_action=((*basic_shape, env_stats.action_dim[aid]), tf.float32, 'prev_action'), 
        )
        TensorSpecs = dict(
            actor_inp=actor_inp,
            value_inp=value_inp,
            evaluation=evaluation,
            return_eval_stats=evaluation,
        )
        if self.has_rnn:
            for inp in [actor_inp, value_inp]:
                inp['mask'] = (basic_shape, tf.float32, 'mask')
            if self.actor_has_rnn:
                TensorSpecs['actor_state'] = self.actor_state_type(
                    *[((None, sz), dtype, name) 
                    for name, sz in self.actor_state_size._asdict().items()])
            if self.value_has_rnn:
                TensorSpecs['value_state'] = self.value_state_type(
                    *[((None, sz), dtype, name) 
                    for name, sz in self.value_state_size._asdict().items()])
        if env_stats.use_action_mask:
            TensorSpecs['actor_inp']['action_mask'] = (
                (*basic_shape, env_stats.action_dim[aid]), tf.bool, 'action_mask'
            )
        if env_stats.use_life_mask:
            TensorSpecs['value_inp']['life_mask'] = (
                basic_shape, tf.float32, 'life_mask'
            )
        self.action = build(self.action, TensorSpecs)

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
                'pi': pi
            })
        return action, terms, state


def create_model(
        config, 
        env_stats, 
        name='ppo', 
        to_build=False,
        to_build_for_eval=False,
        **kwargs):
    aid = config['aid']
    config.policy.policy.action_dim = env_stats.action_dim[aid]
    config.policy.policy.is_action_discrete = env_stats.is_action_discrete[aid]
    config.value.action_embed.input_dim = env_stats.action_dim[aid]
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
