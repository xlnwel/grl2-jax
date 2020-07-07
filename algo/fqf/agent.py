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
        is_nsteps = dataset and 'steps' in dataset.data_format
        self.dataset = dataset

        if self._schedule_lr:
            self._lr = TFPiecewiseSchedule(
                [(5e5, self._lr), (2e6, 5e-5)], outside_value=5e-5)
        if self._schedule_act_eps:
            self._act_eps = PiecewiseSchedule(((5e4, 1), (4e6, self._act_eps)))

        self._to_sync = Every(self._target_update_period)
        self._to_summary = Every(self.LOG_PERIOD)
        # optimizer
        self._iqn_opt = Optimizer(self._iqn_opt, self.q, self._iqn_lr, clip_norm=self._clip_norm)
        self._fpn_opt = Optimizer(self._fpn_opt, self.fpn, self._fpn_lr)

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

    def __call__(self, obs, deterministic=False, **kwargs):
        if self._schedule_act_eps:
            eps = self._act_eps.value(self.env_step)
            self.store(act_eps=eps)
        else:
            eps = self._act_eps
        eps = tf.convert_to_tensor(eps, tf.float32)

        return self.model.action(
            tf.convert_to_tensor(obs), 
            deterministic=deterministic, 
            epsilon=eps).numpy()

    @step_track
    def learn_log(self, step):
        for _ in range(self.N_UPDATES):
            data = self.dataset.sample()

            if self._is_per:
                idxes = data.pop('idxes').numpy()

            terms = self.learn(**data)
            if self._to_sync(self.train_step):
                self._sync_target_nets()
            if self._to_summary(step):
                self.summary(data, terms)
                print(terms['tau'][0])

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
    def summary(self, data, terms):
        tf.summary.histogram('tau', terms['tau'], step=self._env_step)
        tf.summary.histogram('fpn_entropy', terms['fpn_entropy'], step=self._env_step)

    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        terms = {}
        with tf.GradientTape(persistent=True) as tape:
            x = self.q.cnn(obs)
            x_no_grad = tf.stop_gradient(x) # forbid gradients to cnn when computing fpn loss
            
            tau, tau_hat, fpn_entropy = self.fpn(x_no_grad)
            terms['tau'] = tau
            tau_hat = tf.stop_gradient(tau_hat) # forbid gradients to fpn when computing qr loss
            qtv, q = self.q.value(
                x, tau_hat, tau_range=tau, action=action)
            next_x = self.q.cnn(next_obs)
            
            next_tau, next_tau_hat, _ = self.fpn(next_x)
            next_action = self.q.action(next_x, next_tau_hat, tau_range=next_tau)
            
            next_x = self.target_q.cnn(next_obs)
            next_tau, next_tau_hat, _ = self.fpn(next_x)
            next_qtv, _ = self.target_q.value(
                next_x, next_tau_hat, tau_range=next_tau, action=next_action)
            tf.debugging.assert_shapes([[tau, (None, self.N+1)]])
            tf.debugging.assert_shapes([[tau_hat, (None, self.N)]])
            tf.debugging.assert_shapes([[qtv, (None, self.N, 1)]])
            tf.debugging.assert_shapes([[q, (None, )]])
            
            reward = reward[:, None, None]
            discount = discount[:, None, None]
            if not isinstance(steps, int):
                steps = steps[:, None, None]
            returns = n_step_target(reward, next_qtv, discount, self._gamma, steps, self._tbo)
            tf.debugging.assert_shapes([[returns, (None, self.N, 1)]])
            returns = tf.transpose(returns, (0, 2, 1))      # [B, 1, N]
            returns = tf.stop_gradient(returns)

            error = returns - qtv   # [B, N, N]
            
            # loss
            tau_hat = tf.expand_dims(tau_hat, -1) # [B, N, 1]
            tf.debugging.assert_shapes([[tau_hat, (None, self.N, 1)]])
            tf.debugging.assert_shapes([[qtv, (None, self.N, 1)]])
            tf.debugging.assert_shapes([[returns, (None, 1, self.N)]])
            tf.debugging.assert_shapes([[error, (None, self.N, self.N)]])

            weight = tf.abs(tau_hat - tf.cast(error < 0, tf.float32))        # [B, N, N']
            huber = huber_loss(error, threshold=self.KAPPA)             # [B, N, N']
            qr_loss = tf.reduce_sum(tf.reduce_mean(weight * huber, axis=2), axis=1) # [B]
            qr_loss = tf.reduce_mean(qr_loss)

            # compute gradients for fpn
            tau_qtv = self.q.value(x_no_grad, tau[..., 1:-1], action=action)     # [B, N-1, A]
            qtv = tf.squeeze(qtv, -1)
            tau_qtv = tf.squeeze(tau_qtv, -1)
            tf.debugging.assert_shapes([
                [qtv, (None, self.N)],
                [tau_qtv, (None, self.N-1)],
            ])

            # we use 𝜃 to represent F^{-1} for brevity
            diff1 = tau_qtv - qtv[..., :-1]  # 𝜃(𝜏[i]) - 𝜃(\hat 𝜏[i-1])
            sign1 = tau_qtv > qtv[..., :-1]
            abs_diff1 = tf.where(sign1, diff1, -diff1)
            diff2 = tau_qtv - qtv[..., 1:]  # 𝜃(𝜏[i]) - 𝜃(\hat 𝜏[i])
            sign2 = tau_qtv > qtv[..., 1:]
            abs_diff2 = tf.where(sign2, diff2, -diff2)
            fpn_out_grads = abs_diff1 + abs_diff2
            fpn_out_grads = tf.stop_gradient(fpn_out_grads)
            fpn_loss = tf.reduce_mean(fpn_out_grads * tau[..., 1:-1])
            fpn_entropy_loss = -self._ent_coef * tf.reduce_mean(fpn_entropy)
            fpn_loss = fpn_loss + fpn_entropy_loss

        if self._is_per:
            error = tf.reduce_max(tf.reduce_mean(tf.abs(error), axis=2), axis=1)
            priority = self._compute_priority(error)
            terms['priority'] = priority
        
        terms['iqn_norm'] = self._iqn_opt(tape, qr_loss)
        terms['fpn_norm'] = self._fpn_opt(tape, fpn_loss)
        
        terms.update(dict(
            q=q,
            returns=returns,
            qr_loss=qr_loss,
            fpn_entropy=fpn_entropy,
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
