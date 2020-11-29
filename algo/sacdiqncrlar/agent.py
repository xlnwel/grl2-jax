import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.rl_utils import n_step_target, quantile_regression_loss
from utility.tf_utils import explained_variance
from utility.schedule import TFPiecewiseSchedule
from core.optimizer import Optimizer
from algo.dqn.base import DQNBase


def get_data_format(env, is_per=False, n_steps=1, dtype=tf.float32):
    obs_dtype = env.obs_dtype if len(env.obs_shape) == 3 else dtype
    action_dtype = tf.int32 if env.is_action_discrete else dtype
    data_format = dict(
        obs=((None, *env.obs_shape), obs_dtype),
        action=((None, *env.action_shape), action_dtype),
        ar=((None, ), tf.int32),
        reward=((None, ), dtype), 
        next_obs=((None, *env.obs_shape), obs_dtype),
        discount=((None, ), dtype),
    )
    if is_per:
        data_format['IS_ratio'] = ((None, ), dtype)
        data_format['idxes'] = ((None, ), tf.int32)
    if n_steps > 1:
        data_format['steps'] = ((None, ), dtype)

    return data_format

def out_op(x, y, op):
    B, D1 = x.shape
    D2 = y.shape[1]
    x = tf.expand_dims(x, -1)   # [B, D1, 1]
    y = tf.expand_dims(y, 1)    # [B, 1, D2]
    z = op(x, y)                # [B, D1, D2]
    z = tf.reshape(z, (B, D1*D2))
    return z


class Agent(DQNBase):
    def _construct_optimizers(self):
        if self._schedule_lr:
            self._actor_lr = TFPiecewiseSchedule([(4e6, self._actor_lr), (7e6, 1e-5)])
            self._q_lr = TFPiecewiseSchedule([(4e6, self._q_lr), (7e6, 1e-5)])

        self._actor_opt = Optimizer(self._optimizer, self.actor, self._actor_lr)
        q_models = [self.encoder, self.q]
        self._twin_q = hasattr(self, 'q2')
        if self._twin_q:
            q_models.append(self.q2)
        self._q_opt = Optimizer(self._optimizer, q_models, self._q_lr)
        self._crl_opt = Optimizer(self._optimizer, [self.encoder, self.crl], self._crl_lr)
        if isinstance(self.temperature, float):
            self.temperature = tf.Variable(self.temperature, trainable=False)
        else:
            self._temp_opt = Optimizer(self._optimizer, self.temperature, self._temp_lr)

    def _add_attributes(self):
        self._max_ar = self.actor.max_ar
        self._combined_action_dim = self._action_dim * self._max_ar

    def __call__(self, x, evaluation=False, **kwargs):
        if evaluation:
            eps = self._eval_act_eps
        elif self._schedule_act_eps:
            eps = self._act_eps.value(self.env_step)
            self.store(act_eps=eps)
        else:
            eps = self._act_eps
        action, ar, terms = self.model.action(
            tf.convert_to_tensor(x), 
            deterministic=evaluation, 
            epsilon=tf.convert_to_tensor(eps, tf.float32))
        action = action.numpy()
        ar = ar.numpy()

        return action, ar, {'ar': ar}

    def _build_learn(self, env):
        # Explicitly instantiate tf.function to initialize variables
        obs_dtype = env.obs_dtype if len(env.obs_shape) == 3 else tf.float32
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=((), tf.int32, 'action'),
            ar=((), tf.int32, 'ar'),
            reward=((), tf.float32, 'reward'),
            next_obs=(env.obs_shape, env.obs_dtype, 'next_obs'),
            discount=((), tf.float32, 'discount'),
        )
        if self._is_per:
            TensorSpecs['IS_ratio'] = ((), tf.float32, 'IS_ratio')
        if self._n_steps > 1:
            TensorSpecs['steps'] = ((), tf.float32, 'steps')
        from core.tf_config import build
        self.learn = build(self._learn, TensorSpecs, batch_size=self._batch_size)

    @tf.function
    def summary(self, data, terms):
        tf.summary.histogram('entropy', terms['entropy'], step=self._env_step)
        tf.summary.histogram('next_act_probs', terms['next_act_probs'], step=self._env_step)
        tf.summary.histogram('next_act_logps', terms['next_act_logps'], step=self._env_step)
        tf.summary.histogram('reward', data['reward'], step=self._env_step)

    @tf.function
    def _learn(self, obs, action, ar, reward, next_obs, discount, steps=1, IS_ratio=1):
        terms = {}
        if not hasattr(self, '_target_entropy'):
            # Entropy of a uniform distribution
            self._target_entropy = np.log(self._combined_action_dim)
            self._target_entropy *= self._target_entropy_coef
        # compute target returns
        next_x = self.encoder(next_obs)
        next_act_probs, next_act_logps, next_ar_probs, next_ar_logps = \
            self.actor.train_step(next_x)
        next_pi_probs = out_op(next_act_probs, next_ar_probs, tf.multiply)
        next_pi_logps = out_op(next_act_logps, next_ar_logps, tf.add)
        next_pi_probs_ed = tf.expand_dims(next_pi_probs, axis=1)  # [B, 1, A*AR]
        next_pi_logps_ed = tf.expand_dims(next_pi_logps, axis=1)  # [B, 1, A*AR]
        next_x = self.target_encoder(next_obs)
        _, next_qtv = self.target_q(next_x, self.N_PRIME)
        if self._twin_q:
            _, next_qtv2 = self.target_q2(next_x, self.N_PRIME)
            next_qtv = (next_qtv + next_qtv2) / 2.
        tf.debugging.assert_shapes([
            [next_act_probs, (None, self._action_dim)],
            [next_act_logps, (None, self._action_dim)],
            [next_ar_probs, (None, self._max_ar)],
            [next_ar_logps, (None, self._max_ar)],
            [next_pi_probs, (None, self._combined_action_dim)],
            [next_pi_logps, (None, self._combined_action_dim)],
            [next_qtv, (None, self.N_PRIME, self._combined_action_dim)],
        ])
        if isinstance(self.temperature, (tf.Variable)):
            temp = self.temperature
        else:
            _, temp = self.temperature()
        reward = reward[:, None]
        discount = discount[:, None]
        if not isinstance(steps, int):
            steps = steps[:, None]
        next_state_qtv = tf.reduce_sum(next_pi_probs_ed 
            * (next_qtv - temp * next_pi_logps_ed), axis=-1)
        returns = n_step_target(reward, next_state_qtv, discount, self._gamma, steps)
        returns = tf.expand_dims(returns, axis=1)      # [B, 1, N']

        tf.debugging.assert_shapes([
            [next_state_qtv, (None, self.N_PRIME)],
            [next_pi_probs_ed, (None, 1, self._combined_action_dim)],
            [next_pi_logps_ed, (None, 1, self._combined_action_dim)],
            [next_qtv, (None, self.N_PRIME, self._combined_action_dim)],
            [returns, (None, 1, self.N_PRIME)],
        ])
        with tf.GradientTape(persistent=True) as tape:
            x = self.encoder(obs)

            q_action = action * self._max_ar + ar
            q_action = tf.one_hot(q_action, self._combined_action_dim)
            q_action_ed = tf.expand_dims(q_action, axis=1)

            qs, error, qr_loss = self._compute_qr_loss(self.q, x, q_action_ed, returns, IS_ratio)
            if self._twin_q:
                qs2, error2, qr_loss2 = self._compute_qr_loss(self.q2, x, q_action_ed, returns, IS_ratio)
                qs = (qs + qs2) / 2.
                error = (error + error2) / 2.
                qr_loss = (qr_loss + qr_loss2) / 2.
            
            with tape.stop_recording():
                x_pos = self.target_encoder(obs)
                z_pos = self.crl(x_pos)
            z_anchor = self.crl(x)
            logits = self.crl.logits(z_anchor, z_pos)
            tf.debugging.assert_shapes([[logits, (self._batch_size, self._batch_size)]])
            labels = tf.range(self._batch_size)
            infonce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
            infonce = tf.reduce_mean(infonce)
            crl_loss = self._crl_coef * infonce
        terms['q_norm'] = self._q_opt(tape, qr_loss)
        terms['crl_norm'] = self._crl_opt(tape, crl_loss)

        action = tf.one_hot(action, self._action_dim)
        with tf.GradientTape() as tape:
            act_probs, act_logps, ar_probs, ar_logps = \
                self.actor.train_step(x)
            pi_probs = out_op(act_probs, ar_probs, tf.multiply)
            pi_logps = out_op(act_logps, ar_logps, tf.add)
            q = tf.reduce_sum(pi_probs * qs, axis=-1)
            entropy = - tf.reduce_sum(pi_probs * pi_logps, axis=-1)
            actor_loss = -(q + temp * entropy)
            tf.debugging.assert_shapes([[actor_loss, (None, )]])
            actor_loss = tf.reduce_mean(IS_ratio * actor_loss)
        terms['actor_norm'] = self._actor_opt(tape, actor_loss)

        if not isinstance(self.temperature, (float, tf.Variable)):
            with tf.GradientTape() as tape:
                log_temp, temp = self.temperature()
                temp_loss = -log_temp * (self._target_entropy - entropy)
                tf.debugging.assert_shapes([[temp_loss, (None, )]])
                temp_loss = tf.reduce_mean(IS_ratio * temp_loss)
            terms['target_entropy'] = self._target_entropy
            terms['entropy_diff'] = self._target_entropy - entropy
            terms['log_temp'] = log_temp
            terms['temp'] = temp
            terms['temp_loss'] = temp_loss
            terms['temp_norm'] = self._temp_opt(tape, temp_loss)

        if self._is_per:
            error = tf.reduce_max(tf.reduce_mean(error, axis=-1), axis=-1)
            priority = self._compute_priority(error / 2.)
            terms['priority'] = priority
            
        target_q = tf.reduce_mean(returns, axis=-1)
        target_q = tf.squeeze(target_q)
        terms.update(dict(
            ar=ar,
            ar_probs0=ar_probs[:, 0],
            ar_probs1=ar_probs[:, 1],
            ar_probs2=ar_probs[:, 2],
            ar_probs3=ar_probs[:, 3],
            logits_max=tf.reduce_max(logits),
            logits_min=tf.reduce_min(logits),
            infonce=infonce,
            act_probs=act_probs,
            max_act_probs=tf.reduce_max(act_probs),
            actor_loss=actor_loss,
            q=q,
            next_value=tf.reduce_mean(next_state_qtv, axis=-1),
            logpi=act_logps,
            entropy=entropy,
            max_entropy=tf.reduce_max(entropy),
            min_entropy=tf.reduce_min(entropy),
            next_act_probs=next_act_probs,
            next_act_logps=next_act_logps,
            returns=returns,
            qr_loss=qr_loss,
            crl_loss=crl_loss, 
            explained_variance1=explained_variance(target_q, q),
        ))

        return terms

    def _compute_qr_loss(self, q, x, action, returns, IS_ratio):
        tau_hat, qtvs, qs = q(x, self.N, return_q=True)
        qtv = tf.reduce_sum(qtvs * action, axis=-1, keepdims=True)  # [B, N, 1]
        error, qr_loss = quantile_regression_loss(qtv, returns, tau_hat, kappa=self.KAPPA, return_error=True)
            
        tf.debugging.assert_shapes([
            [qtvs, (None, self.N, self._combined_action_dim)],
            [action, (None, 1, self._combined_action_dim)],
            [qtv, (None, self.N, 1)],
            [qs, (None, self._combined_action_dim)],
            [qr_loss, (None)],
        ])

        qr_loss = tf.reduce_mean(IS_ratio * qr_loss)

        return qs, tf.abs(error), qr_loss

    @tf.function
    def _sync_target_nets(self):
        tvars = self.target_encoder.variables + self.target_q.variables
        mvars = self.encoder.variables + self.q.variables
        if self._twin_q:
            tvars += self.target_q2.variables
            mvars += self.q2.variables
        [tvar.assign(mvar) for tvar, mvar in zip(tvars, mvars)]
