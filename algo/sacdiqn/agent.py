import tensorflow as tf

from utility.rl_loss import n_step_target, quantile_regression_loss
from utility.tf_utils import explained_variance
from core.decorator import override
from algo.dqn.base import DQNBase, get_data_format, collect
from algo.sacd.base import TempLearner
from algo.iqn.base import IQNOps


class Agent(DQNBase, IQNOps, TempLearner):
    """ Initialization """
    # @tf.function
    # def summary(self, data, terms):
    #     tf.summary.histogram('learn/entropy', terms['entropy'], step=self._env_step)
    #     tf.summary.histogram('learn/reward', data['reward'], step=self._env_step)
    
    """ Call """
    def _process_input(self, obs, evaluation, env_output):
        obs, kwargs = super()._process_input(obs, evaluation, env_output)
        return obs, kwargs

    """ SACIQN Methods"""
    @override(DQNBase)
    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        if self.temperature.type == 'schedule':
            _, temp = self.temperature(self._train_step)
        elif self.temperature.type == 'state':
            raise NotImplementedError
        else:
            _, temp = self.temperature()

        target, terms = self._compute_target(reward, next_obs, discount, steps, temp)

        with tf.GradientTape() as tape:
            x = self.encoder(obs, training=True)
            tau_hat, qt_embed = self.quantile(x, self.N)
            x_ext = tf.expand_dims(x, axis=1)
            action_ext = tf.expand_dims(action, axis=1)
            # q loss
            qtvs, qs = self.q(x_ext, qt_embed, return_value=True)
            qtv = tf.reduce_sum(qtvs * action_ext, axis=-1, keepdims=True)  # [B, N, 1]
            error, qr_loss = quantile_regression_loss(
                qtv, target, tau_hat, kappa=self.KAPPA, return_error=True)
            value_loss = tf.reduce_mean(IS_ratio * qr_loss)
        terms['value_norm'] = self._value_opt(tape, value_loss)

        with tf.GradientTape() as tape:
            act_probs, act_logps = self.actor.train_step(x)
            q = tf.reduce_sum(act_probs * qs, axis=-1)
            entropy = - tf.reduce_sum(act_probs * act_logps, axis=-1)
            actor_loss = -(q + temp * entropy)
            actor_loss = tf.reduce_mean(IS_ratio * actor_loss)
        terms['actor_norm'] = self._actor_opt(tape, actor_loss)

        act_probs = tf.reduce_mean(act_probs, 0)
        self.actor.update_prior(act_probs, self._prior_lr)

        temp_terms = self._learn_temp(x, entropy, IS_ratio)

        if self._is_per:
            error = tf.abs(error)
            error = tf.reduce_max(tf.reduce_mean(error, axis=-1), axis=-1)
            priority = self._compute_priority(error)
            terms['priority'] = priority
            
        target = tf.reduce_mean(target, axis=(1, 2))
        terms.update(dict(
            IS_ratio=IS_ratio,
            steps=steps,
            reward_min=tf.reduce_min(reward),
            actor_loss=actor_loss,
            q=q,
            entropy=entropy,
            entropy_max=tf.reduce_max(entropy),
            entropy_min=tf.reduce_min(entropy),
            value_loss=value_loss, 
            temp=temp,
            explained_variance_q=explained_variance(target, q),
            **temp_terms
        ))
        # for i in range(self.actor.action_dim):
        #     terms[f'prior_{i}'] = self.actor.prior[i]

        return terms

    def _compute_target(self, reward, next_obs, discount, steps, temp):
        terms = {}

        next_x = self.target_encoder(next_obs, training=False)
        next_act_probs, next_act_logps = self.target_actor.train_step(next_x)
        next_act_probs_ext = tf.expand_dims(next_act_probs, axis=1)  # [B, 1, A]
        next_act_logps_ext = tf.expand_dims(next_act_logps, axis=1)  # [B, 1, A]
        _, qt_embed = self.target_quantile(next_x, self.N_PRIME)
        next_x_ext = tf.expand_dims(next_x, axis=1)
        next_qtv = self.target_q(next_x_ext, qt_embed)
        
        if self._soft_target:
            next_qtv_v = tf.reduce_sum(next_act_probs_ext 
                * (next_qtv - temp * next_act_logps_ext), axis=-1)
        else:
            next_qtv_v = tf.reduce_sum(next_act_probs_ext * next_qtv, axis=-1)
        reward = reward[:, None]
        discount = discount[:, None]
        if not isinstance(steps, int):
            steps = steps[:, None]
        target = n_step_target(reward, next_qtv_v, discount, self._gamma, steps)
        target = tf.expand_dims(target, axis=1)      # [B, 1, N']
        tf.debugging.assert_shapes([
            [next_qtv_v, (None, self.N_PRIME)],
            [next_act_probs_ext, (None, 1, self._action_dim)],
            [next_act_logps_ext, (None, 1, self._action_dim)],
            [target, (None, 1, self.N_PRIME)],
        ])
        return target, terms
