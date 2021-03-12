import tensorflow as tf

from utility.tf_utils import explained_variance
from utility.rl_loss import ppo_loss
from core.tf_config import build
from core.decorator import override
from algo.ppo.base import PPOBase


class Agent(PPOBase):
    """ Initialization """
    @override(PPOBase)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=(env.action_shape, env.action_dtype, 'action'),
            value=((), tf.float32, 'value'),
            traj_ret=((), tf.float32, 'traj_ret'),
            advantage=((), tf.float32, 'advantage'),
            logpi=((), tf.float32, 'logpi'),
        )
        self.learn = build(self._learn, TensorSpecs)

    # @override(PPOBase)
    # def _summary(self, data, terms):
    #     tf.summary.histogram('sum/value', data['value'], step=self._env_step)
    #     tf.summary.histogram('sum/logpi', data['logpi'], step=self._env_step)

    @tf.function
    def _learn(self, obs, action, value, traj_ret, advantage, logpi, state=None, mask=None, additional_input=[]):
        old_value = value
        terms = {}
        with tf.GradientTape() as tape:
            x = self.encoder(obs)
            if state is not None:
                x, _ = self.rnn(x, state, mask=mask, additional_input=additional_input)
            prior_dist = self.prior(x)
            act_dist = self.actor(x)
            new_logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            # policy loss
            log_ratio = new_logpi - logpi
            policy_loss, entropy, kl, p_clip_frac = ppo_loss(
                log_ratio, advantage, self._clip_range, entropy)
            # value loss
            value = self.value(x)
            value_loss, v_clip_frac = self._compute_value_loss(value, traj_ret, old_value)
            
            prior_kl = tf.reduce_mean(act_dist.kl_divergence(prior_dist))
            actor_loss = (policy_loss - self._entropy_coef * entropy + self._kl_coef * prior_kl)
            value_loss = self._value_coef * value_loss
            ac_loss = actor_loss + value_loss

        terms['ac_norm'] = self._optimizer(tape, ac_loss)
        terms.update(dict(
            value=value,
            traj_ret=tf.reduce_mean(traj_ret), 
            advantage=tf.reduce_mean(advantage), 
            ratio=tf.exp(log_ratio), 
            entropy=entropy, 
            kl=kl, 
            prior_kl=prior_kl,
            p_clip_frac=p_clip_frac,
            ppo_loss=policy_loss,
            actor_loss=actor_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value),
            v_clip_frac=v_clip_frac
        ))

        return terms