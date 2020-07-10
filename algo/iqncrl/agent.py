import numpy as np
import tensorflow as tf

from utility.display import pwc
from utility.rl_utils import n_step_target, huber_loss
from utility.utils import Every
from utility.schedule import TFPiecewiseSchedule, PiecewiseSchedule
from utility.timer import TBTimer
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config, step_track
from core.optimizer import Optimizer
from algo.dqn.agent import get_data_format


class Agent(BaseAgent):
    @agent_config
    def __init__(self, *, dataset, env):
        self._is_per = self._replay_type.endswith('per')
        is_nsteps = self._n_steps > 1
        self.dataset = dataset

        if self._schedule_lr:
            self._lr = TFPiecewiseSchedule(
                [(5e5, self._lr), (2e6, 5e-5)], outside_value=5e-5)
        if self._schedule_act_eps:
            self._act_eps = PiecewiseSchedule(((5e4, 1), (4e6, self._act_eps)))

        self._to_sync = Every(self._target_update_period)
        # optimizer
        self._optimizer = Optimizer(self._optimizer, [self.q, self.crl], self._lr, clip_norm=self._clip_norm)

        self._action_dim = env.action_dim

        # Explicitly instantiate tf.function to initialize variables
        obs_dtype = env.obs_dtype if len(env.obs_shape) == 3 else self._dtype
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=((self._action_dim,), tf.float32, 'action'),
            reward=((), self._dtype, 'reward'),
            next_obs=(env.obs_shape, env.obs_dtype, 'next_obs'),
            discount=((), self._dtype, 'discount'),
        )
        if self._is_per:
            TensorSpecs['IS_ratio'] = ((), self._dtype, 'IS_ratio')
        if is_nsteps:
            TensorSpecs['steps'] = ((), self._dtype, 'steps')
        self.learn = build(self._learn, TensorSpecs, batch_size=self._batch_size)

        self._sync_target_nets()

        self._to_summary = Every(self.LOG_PERIOD, self.LOG_PERIOD)

    def reset_noisy(self):
        pass

    def __call__(self, x, deterministic=False, **kwargs):
        if self._schedule_act_eps:
            eps = self._act_eps.value(self.env_step)
            self.store(act_eps=eps)
        else:
            eps = self._act_eps

        action, terms = self.model.action(
            tf.convert_to_tensor(x), 
            deterministic=deterministic, 
            epsilon=eps)
        action = action.numpy()

        return action

    @step_track
    def learn_log(self, step):
        for _ in range(self.N_UPDATES):
            data = self.dataset.sample()

            if self._is_per:
                idxes = data.pop('idxes').numpy()

            terms = self.learn(**data)
            if self._to_sync(self.train_step):
                self._sync_target_nets()

            if self._schedule_lr:
                step = tf.convert_to_tensor(step, tf.float32)
                terms['lr'] = self._lr(step)
            terms = {k: v.numpy() for k, v in terms.items()}

            if self._is_per:
                self.dataset.update_priorities(terms['priority'], idxes)
            self.store(**terms)
        if self._to_summary(self.train_step):
            self.summary(data)

        return self.N_UPDATES

    @tf.function
    def summary(self, data):
        tf.summary.histogram('stats/steps', data['steps'], step=self._env_step)
        if 'IS_ratio' in data:
            self.histogram_summary({'IS_ratio': data['IS_ratio']}, step=self._env_step)

    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        terms = {}
        with tf.GradientTape() as tape:
            z, tau_hat, qtv, q = self.q.z_value(obs, self.N, action)
            next_action = self.q.action(next_obs, self.K)
            _, next_qtv, _ = self.target_q.value(next_obs, self.N_PRIME, next_action)
            reward = reward[None, :, None]
            discount = discount[None, :, None]
            if not isinstance(steps, int):
                steps = steps[None, :, None]
            returns = n_step_target(reward, next_qtv, discount, self._gamma, steps, self._tbo)
            qtv = tf.transpose(qtv, (1, 0, 2))              # [B, N, 1]
            returns = tf.transpose(returns, (1, 2, 0))      # [B, 1, N']
            returns = tf.stop_gradient(returns)

            error = returns - qtv   # [B, N, N']
            
            # loss
            tau_hat = tf.transpose(tf.reshape(tau_hat, [self.N, self._batch_size, 1]), [1, 0, 2]) # [B, N, 1]
            weight = tf.abs(tau_hat - tf.cast(error < 0, tf.float32))        # [B, N, N']
            huber = huber_loss(error, threshold=self.KAPPA)             # [B, N, N']
            qr_loss = tf.reduce_sum(tf.reduce_mean(weight * huber, axis=2), axis=1) # [B]
            qr_loss = tf.reduce_mean(qr_loss)

            z_pos = self.target_q.cnn(obs)
            z_anchor = self.crl(z)
            z_pos = self.crl(z_pos)
            logits = self.crl.logits(z_anchor, z_pos)
            tf.debugging.assert_shapes([[logits, (self._batch_size, self._batch_size)]])
            labels = tf.range(self._batch_size)
            infonce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
            infonce = tf.reduce_mean(infonce)
            loss = qr_loss + self._crl_coef * infonce

        if self._is_per:
            error = tf.reduce_max(tf.reduce_mean(tf.abs(error), axis=2), axis=1)
            priority = self._compute_priority(error)
            terms['priority'] = priority
        
        terms['norm'] = self._optimizer(tape, loss)
        
        terms.update(dict(
            q=q,
            returns=returns,
            qr_loss=qr_loss,
            infonce=infonce,
            loss=loss,
        ))

        return terms

    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        priority += self._per_epsilon
        priority **= self._per_alpha
        return priority

    @tf.function
    def _sync_target_nets(self):
        [tv.assign(mv) for mv, tv in zip(
            self.q.variables, self.target_q.variables)]
