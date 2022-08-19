import os
import tensorflow as tf

from utility.file import source_file
from .utils import compute_inner_steps, get_hx
from algo.zero.elements.model import Model as ModelBase, ModelEnsemble
# register ppo-related networks 
source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class Model(ModelBase):
    def compute_eval_terms(self, global_state, hidden_state, action, hx):
        value = self.value(global_state, hx=hx)
        value = tf.squeeze(value)
        meta_reward, trans_reward = self.compute_meta_reward(hidden_state, action, hx=hx)
        meta_reward = tf.squeeze(meta_reward)
        trans_reward = tf.squeeze(trans_reward)
        terms = {'meta_reward': meta_reward, 'trans_reward': trans_reward, 
            'value': value, 'action': action}
        return terms

    def compute_meta_reward(self, hidden_state, action, idx=None, event=None, hx=None):
        action_oh = tf.one_hot(action, self.policy.action_dim)
        x = tf.concat([hidden_state, action_oh], -1)
        if hx is None:
            hx = get_hx(idx, event)
        meta_reward = self.meta_reward(x, hx=hx)
        reward_scale = self.meta('reward_scale', inner=True)
        reward_bias = self.meta('reward_bias', inner=True)
        reward = reward_scale * meta_reward + reward_bias

        return meta_reward, reward


def setup_config_from_envstats(config, env_stats):
    if 'aid' in config:
        aid = config['aid']
        config.policy.action_dim = env_stats.action_dim[aid]
        config.policy.is_action_discrete = env_stats.is_action_discrete[aid]
        config.meta.reward_scale.shape = len(env_stats.aid2uids[aid])
        config.meta.reward_bias.shape = len(env_stats.aid2uids[aid])
        config.meta.reward_coef.shape = len(env_stats.aid2uids[aid])
    else:
        config.policy.action_dim = env_stats.action_dim
        config.policy.is_action_discrete = env_stats.is_action_discrete
        config.policy.action_low = env_stats.get('action_low')
        config.policy.action_high = env_stats.get('action_high')
        config.meta.reward_scale.shape = env_stats.n_units
        config.meta.reward_bias.shape = env_stats.n_units
        config.meta.reward_coef.shape = env_stats.n_units
    return config


def create_model(
    config, 
    env_stats, 
    name='zero', 
    to_build=False, 
    to_build_for_eval=False, 
    **kwargs
): 
    config = setup_config_from_envstats(config, env_stats)
    config = compute_inner_steps(config)

    if config['rnn_type'] is None:
        config.pop('rnn', None)
    else:
        config['rnn']['nn_id'] = config['actor_rnn_type']

    build_meta = config.inner_steps == 1 and config.extra_meta_step == 0
    rl = Model(
        config=config, 
        env_stats=env_stats, 
        name='rl',
        to_build=not build_meta and to_build, 
        to_build_for_eval=not build_meta and to_build_for_eval,
        **kwargs
    )
    meta = Model(
        config=config, 
        env_stats=env_stats, 
        name='meta',
        to_build=build_meta and to_build, 
        to_build_for_eval=build_meta and to_build_for_eval,
        **kwargs
    )
    return ModelEnsemble(
        config=config, 
        env_stats=env_stats, 
        components=dict(
            rl=rl, 
            meta=meta, 
        ), 
        name=name, 
        to_build=to_build, 
        to_build_for_eval=to_build_for_eval,
        **kwargs
    )
