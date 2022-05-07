import os
import tensorflow as tf

from utility.file import source_file
from algo.zero.elements.model import PPOValueModel as PPOValueBase, \
    PPOActorModel, PPOModelEnsemble

source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class PPOValueModel(PPOValueBase):
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
        ae_weights = self.hyper_ae(global_state)
        eqt = 'bah,baho->bao' if len(raw_ae.shape) == 3 else 'btah,btaho->btao'
        ae = tf.einsum(eqt, raw_ae, ae_weights)

        return ae


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
