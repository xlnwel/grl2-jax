import os
import collections
import tensorflow as tf

from core.tf_config import build
from utility.file import source_file
from algo.hm.elements.model import ModelImpl, MAPPOActorModel, \
    MAPPOModelEnsemble as MAPPOModelBase

source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class MAPPOValueModel(ModelImpl):
    def compute_value(
        self, 
        global_state, 
        action, 
        prev_reward, 
        prev_action, 
        state=None, 
        mask=None
    ):
        ae = self.action_embed(action)
        x_a = tf.concat([global_state, ae], -1)
        x = tf.concat([global_state, tf.zeros_like(ae)], -1)

        x, state = self.encode(
            x, 
            prev_reward, 
            prev_action, 
            state, 
            mask
        )

        value = self.value(x)

        x_a, state = self.encode(
            x_a, 
            prev_reward, 
            prev_action, 
            state, 
            mask
        )
        value_a = self.value(x_a)

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
        if self.has_rnn:
            actor_inp, value_inp = self._add_seqential_dimension(
                actor_inp, value_inp)
        action, terms, actor_state = self.policy.action(
            **actor_inp, state=actor_state,
            evaluation=evaluation, return_eval_stats=return_eval_stats)
        value, value_a, value_state = self.value.compute_value(
            **value_inp, action=action, state=value_state)
        state = self.state_type(*actor_state, *value_state) \
            if self.has_rnn else None
        if not evaluation:
            terms.update({
                'value': value,
                'value_a': value_a,
            })
        if self.has_rnn:
            action, terms = self._remove_seqential_dimension(action, terms)
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
    config.value.action_embed.action_dim = env_stats.action_dim[aid]
    config.value.action_embed.n_units = env_stats.n_units

    if config.get('rnn_type') is None:
        for c in config.values():
            if isinstance(c, dict):
                c.pop('rnn', None)
    else:
        for c in config.values():
            if isinstance(c, dict) and 'rnn' in c:
                c['rnn']['nn_id'] = config['rnn_type']

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
