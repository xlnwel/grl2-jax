import os
import tensorflow as tf

from core.log import do_logging
from core.tf_config import build
from utility.file import source_file
from .utils import compute_inner_steps, get_hx
from algo.zero.elements.model import Model as ModelBase, ModelEnsemble

# register ppo-related networks 
source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class Model(ModelBase):
    def _build(
        self, 
        env_stats, 
        evaluation: bool=False
    ):
        aid = self.config.get('aid', 0)
        basic_shape = (None, len(env_stats.aid2uids[aid]))
        dtype = tf.keras.mixed_precision.global_policy().compute_dtype
        shapes = env_stats['obs_shape'][aid]
        dtypes = env_stats['obs_dtype'][aid]
        TensorSpecs = {k: ((*basic_shape, *v), dtypes[k], k) 
            for k, v in shapes.items()}
        TensorSpecs['prev_hidden_state'] = TensorSpecs['hidden_state']
        if 'idx' in TensorSpecs:
            TensorSpecs['prev_idx'] = TensorSpecs['idx']
        if 'event' in TensorSpecs:
            TensorSpecs['prev_event'] = TensorSpecs['event']
        TensorSpecs.update(dict(
            evaluation=evaluation,
            return_eval_stats=evaluation,
        ))
        if self.has_rnn:
            TensorSpecs.update(dict(
                state=self.state_type(*[((None, sz), dtype, name) 
                    for name, sz in self.state_size._asdict().items()]),
                mask=(basic_shape, tf.float32, 'mask'),
            ))
        if env_stats.use_action_mask:
            TensorSpecs['action_mask'] = (
                (*basic_shape, env_stats.action_dim[aid]), tf.bool, 'action_mask'
            )
        if env_stats.use_life_mask:
            TensorSpecs['life_mask'] = (
                basic_shape, tf.float32, 'life_mask'
            )
        do_logging(TensorSpecs, prefix='Tensor Specifications', level='print')
        self.action = build(self.action, TensorSpecs)

    @tf.function
    def action(
        self, 
        obs, 
        idx=None, 
        event=None, 
        global_state=None, 
        hidden_state=None, 
        action_mask=None, 
        life_mask=None, 
        prev_reward=None,
        prev_action=None,
        prev_idx=None, 
        prev_event=None, 
        prev_hidden_state=None, 
        state=None,
        mask: tf.Tensor=None,
        evaluation=False, 
        return_eval_stats=False
    ):
        x, state = self.encode(obs, state=state, mask=mask)
        hx = get_hx(idx, event)
        act_dist = self.policy(
            x, hx=hx, 
            action_mask=action_mask, 
            evaluation=evaluation
        )
        action = self.policy.action(act_dist, evaluation)

        if self.policy.is_action_discrete:
            pi = tf.nn.softmax(act_dist.logits)
            terms = {'mu': pi}
        else:
            mean = act_dist.mean()
            std = tf.exp(self.policy.logstd)
            terms = {
                'mu_mean': mean,
                'mu_std': std * tf.ones_like(mean), 
            }

        if global_state is None:
            global_state = x
        if evaluation:
            if self.config.meta_reward_type == 'shaping':
                prev_hx = get_hx(prev_idx, prev_event)
            else:
                prev_hx = None
            terms = self.compute_eval_terms(
                global_state, 
                prev_hidden_state, 
                hidden_state, 
                action, 
                prev_hx, 
                hx
            )
            
            return action, terms, state
        else:
            logprob = act_dist.log_prob(action)
            tf.debugging.assert_all_finite(logprob, 'Bad logprob')
            value = self.value(global_state, hx=hx)
            terms.update({'mu_logprob': logprob, 'value': value})

            return action, terms, state    # keep the batch dimension for later use

    def compute_eval_terms(
        self, 
        global_state, 
        prev_hidden_state, 
        hidden_state, 
        action, 
        prev_hx, 
        hx
    ):
        value = self.value(global_state, hx=hx)
        value = tf.squeeze(value)
        _, meta_reward, trans_reward = self.compute_meta_reward(
            prev_hidden_state, hidden_state, action, 
            hx=prev_hx, next_hx=hx, shift=True
        )
        meta_reward = tf.squeeze(meta_reward)
        trans_reward = tf.squeeze(trans_reward)
        terms = {'meta_reward': meta_reward, 'trans_reward': trans_reward, 
            'value': value, 'action': action}
        return terms

    def compute_meta_reward(
        self, 
        hidden_state, 
        next_hidden_state, 
        action=None, 
        idx=None, 
        next_idx=None, 
        event=None, 
        next_event=None, 
        hx=None, 
        next_hx=None, 
        shift=False,    # if hidden_state/idx/event are shifted by one step. If so action is at the same step as the next stats
    ):
        if hx is None:
            hx = get_hx(idx, event)
        if self.config.meta_reward_type == 'shaping':
            if next_hx is None:
                next_hx = get_hx(next_idx, next_event)
            phi = self.meta_reward(hidden_state, hx=hx)
            next_phi = self.meta_reward(next_hidden_state, hx=next_hx)
            x, meta_reward = self.config.gamma * next_phi - phi
        elif self.config.meta_reward_type == 'intrinsic':
            x = next_hidden_state if shift else hidden_state
            x, meta_reward = self.meta_reward(x, action, hx=next_hx if shift else hx)
        else:
            raise ValueError(f"Unknown meta rewared type: {self.config.meta_reward_type}")
        reward_scale = self.meta('reward_scale', inner=True)
        reward_bias = self.meta('reward_bias', inner=True)
        reward = reward_scale * meta_reward + reward_bias

        return x, meta_reward, reward


def setup_config_from_envstats(config, env_stats):
    if 'aid' in config:
        aid = config['aid']
        config.policy.action_dim = env_stats.action_dim[aid]
        config.policy.is_action_discrete = env_stats.is_action_discrete[aid]
        config.meta_reward.out_size = env_stats.action_dim[aid]
        config.meta.reward_scale.shape = len(env_stats.aid2uids[aid])
        config.meta.reward_bias.shape = len(env_stats.aid2uids[aid])
        config.meta.reward_coef.shape = len(env_stats.aid2uids[aid])
    else:
        config.policy.action_dim = env_stats.action_dim
        config.policy.is_action_discrete = env_stats.is_action_discrete
        config.policy.action_low = env_stats.get('action_low')
        config.policy.action_high = env_stats.get('action_high')
        config.meta_reward.out_size = env_stats.action_dim
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
