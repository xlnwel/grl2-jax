import tensorflow as tf

from core.tf_config import build
from core.decorator import override
from utility.tf_utils import explained_variance
from utility.rl_loss import v_trace_from_ratio, ppo_loss
from algo.ppo2.agent import Agent as PPOBase


def collect(buffer, env, env_step, reset, next_obs, **kwargs):
    buffer.add(**kwargs)


def get_data_format(*, env, batch_size, sample_size=None,
        store_state=False, state_size=None, **kwargs):
    obs_dtype = tf.uint8 if len(env.obs_shape) == 3 else tf.float32
    action_dtype = tf.int32 if env.is_action_discrete else tf.float32
    data_format = dict(
        obs=((None, sample_size+1, *env.obs_shape), obs_dtype),
        action=((None, sample_size, *env.action_shape), action_dtype),
        reward=((None, sample_size), tf.float32),
        value=((None, sample_size), tf.float32), 
        discount=((None, sample_size), tf.float32),
        logpi=((None, sample_size), tf.float32),
        mask=((None, sample_size+1), tf.float32),
    )
    if store_state:
        dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        data_format.update({
            k: ((batch_size, v), dtype)
                for k, v in state_size._asdict().items()
        })

    return data_format


class Agent(PPOBase):
    """ Initialization """
    @override(PPOBase)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=((self._sample_size+1, *env.obs_shape), env.obs_dtype, 'obs'),
            action=((self._sample_size, *env.action_shape), env.action_dtype, 'action'),
            reward=((self._sample_size,), tf.float32, 'reward'),
            value=((self._sample_size,), tf.float32, 'value'),
            discount=((self._sample_size,), tf.float32, 'discount'),
            logpi=((self._sample_size,), tf.float32, 'logpi'),
            mask=((self._sample_size+1,), tf.float32, 'mask'),
        )
        if self._store_state:
            state_type = type(self.model.state_size)
            TensorSpecs['state'] = state_type(*[((sz, ), self._dtype, name) 
                for name, sz in self.model.state_size._asdict().items()])
        if self._additional_rnn_inputs:
            if 'prev_action' in self._additional_rnn_inputs:
                TensorSpecs['prev_action'] = (
                    (self._sample_size, *env.action_shape), 
                    env.action_dtype, 'prev_action')
            if 'prev_reward' in self._additional_rnn_inputs:
                TensorSpecs['prev_reward'] = (
                    (self._sample_size,), self._dtype, 'prev_reward')    # this reward should be unnormlaized
        self.learn = build(self._learn, TensorSpecs)

    # @override(PPOBase)
    # def _summary(self, data, terms):
    #     tf.summary.histogram('sum/value', data['value'], step=self._env_step)
    #     tf.summary.histogram('sum/logpi', data['logpi'], step=self._env_step)

    @tf.function
    def _learn(self, obs, action, reward, value, 
                discount, logpi, mask=None, state=None, 
                prev_action=None, prev_reward=None):
        old_value = value
        terms = {}
        with tf.GradientTape() as tape:
            if hasattr(self.model, 'rnn'):
                x, state = self.model.encode(obs, state, mask,
                    prev_action=prev_action, prev_reward=prev_reward)
            else:
                x = self.encoder(obs)
            curr_x, _ = tf.split(x, [self._sample_size, 1], 1)
            act_dist = self.actor(curr_x)
            new_logpi = act_dist.log_prob(action)
            log_ratio = new_logpi - logpi
            ratio = tf.exp(log_ratio)
            entropy = act_dist.entropy()
            value = self.value(x)
            value, next_value = value[:, :-1], value[:, 1:]
            discount = self._gamma * discount
            # policy loss
            target, advantage = v_trace_from_ratio(
                reward, value, next_value, ratio, discount, 
                lambda_=self._lambda, c_clip=self._c_clip, 
                rho_clip=self._rho_clip, rho_clip_pg=self._rho_clip_pg, axis=1)
            target = tf.stop_gradient(target)
            advantage = tf.stop_gradient(advantage)
            if self._policy_loss == 'ppo':
                policy_loss, entropy, kl, p_clip_frac = ppo_loss(
                    log_ratio, advantage, self._clip_range, entropy)
            elif self._policy_loss == 'reinforce':
                policy_loss = -tf.reduce_mean(advantage * log_ratio)
                entropy = tf.reduce_mean(entropy)
                kl = tf.reduce_mean(-log_ratio)
                p_clip_frac = 0
            else:
                raise NotImplementedError
            # value loss
            value_loss, v_clip_frac = self._compute_value_loss(
                value, target, old_value)

            actor_loss = (policy_loss - self._entropy_coef * entropy)
            value_loss = self._value_coef * value_loss
            ac_loss = actor_loss + value_loss

        terms['ac_norm'] = self._ac_opt(tape, ac_loss)
        terms.update(dict(
            value=value,
            ratio=ratio, 
            entropy=entropy, 
            kl=kl, 
            p_clip_frac=p_clip_frac,
            ppo_loss=policy_loss,
            actor_loss=actor_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(target, value),
            v_clip_frac=v_clip_frac
        ))

        return terms
