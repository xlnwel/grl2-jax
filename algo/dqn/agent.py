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


def get_data_format(env, is_per=False, n_steps=1, dtype=tf.float32):
    obs_dtype = env.obs_dtype if len(env.obs_shape) == 3 else dtype
    action_dtype = tf.int32 if env.is_action_discrete else dtype
    data_format = dict(
        obs=((None, *env.obs_shape), obs_dtype),
        action=((None, *env.action_shape), action_dtype),
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


class DQNBase(BaseAgent):
    @agent_config
    def __init__(self, *, dataset, env):
        self._is_per = self._replay_type.endswith('per')
        is_nsteps = self._n_steps > 1
        self.dataset = dataset

        if self._schedule_act_eps:
            self._act_eps = PiecewiseSchedule(((5e4, 1), (1e6, self._act_eps)))

        self._to_sync = Every(self._target_update_period)
        self._to_summary = Every(self.LOG_PERIOD, self.LOG_PERIOD)

        self._construct_optimizers()

        self._action_dim = env.action_dim

        # Explicitly instantiate tf.function to initialize variables
        obs_dtype = env.obs_dtype if len(env.obs_shape) == 3 else tf.float32
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=((self._action_dim,), tf.float32, 'action'),
            reward=((), tf.float32, 'reward'),
            next_obs=(env.obs_shape, env.obs_dtype, 'next_obs'),
            discount=((), tf.float32, 'discount'),
        )
        if self._is_per:
            TensorSpecs['IS_ratio'] = ((), tf.float32, 'IS_ratio')
        if self._n_steps > 1:
            TensorSpecs['steps'] = ((), tf.float32, 'steps')
        self.learn = build(self._learn, TensorSpecs, batch_size=self._batch_size)

        self._sync_target_nets()


    def reset_noisy(self):
        pass

    def __call__(self, x, deterministic=False, **kwargs):
        if deterministic:
            eps = self._eval_act_eps
        elif self._schedule_act_eps:
            eps = self._act_eps.value(self.env_step)
            self.store(act_eps=eps)
        else:
            eps = self._act_eps
        action, terms = self.model.action(
            tf.convert_to_tensor(x), 
            deterministic=deterministic, 
            epsilon=tf.convert_to_tensor(eps, tf.float32))
        action = np.squeeze(action.numpy())

        return action

    @step_track
    def learn_log(self, step):
        for _ in range(self.N_UPDATES):
            with TBTimer('sample', 2500):
                data = self.dataset.sample()

            if self._is_per:
                idxes = data.pop('idxes').numpy()

            with TBTimer('learn', 2500):
                terms = self.learn(**data)
            if self._to_sync(self.train_step):
                self._sync_target_nets()
            if self._to_summary(step):
                self.summary(data, terms)

            terms = {k: v.numpy() for k, v in terms.items()}

            if self._is_per:
                self.dataset.update_priorities(terms['priority'], idxes)
            self.store(**terms)
        return self.N_UPDATES

    def _construct_optimizers(self):
        raise NotImplementedError

    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        raise NotImplementedError

    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        priority += self._per_epsilon
        priority **= self._per_alpha
        return priority

    def summary(self, data, terms):
        pass

    @tf.function
    def _sync_target_nets(self):
        [tv.assign(mv) for mv, tv in zip(
            self.q.variables, self.target_q.variables)]

class Agent(DQNBase):
    def _construct_optimizers(self):
        if self._schedule_lr:
            self._lr = TFPiecewiseSchedule(
                [(5e5, self._lr), (2e6, 5e-5)], outside_value=5e-5)
        self._optimizer = Optimizer(self._optimizer, self.q, self._lr, clip_norm=self._clip_norm)

    def reset_noisy(self):
        self.q.reset_noisy()

    @tf.function
    def _learn(self, obs, action, reward, next_obs, discount, steps=1, IS_ratio=1):
        loss_fn = dict(
            huber=huber_loss, mse=lambda x: .5 * x**2)[self._loss_type]
        terms = {}
        # compute target returns
        next_action = self.q.action(next_obs, noisy=False)
        next_q = self.target_q.value(next_obs, next_action, noisy=False)
        returns = n_step_target(reward, next_q, discount, self._gamma, steps, self._tbo)

        with tf.GradientTape() as tape:
            q = self.q.value(obs, action)
            error = returns - q
            loss = tf.reduce_mean(IS_ratio * loss_fn(error))

        if self._is_per:
            priority = self._compute_priority(tf.abs(error))
            terms['priority'] = priority
        
        terms['norm'] = self._optimizer(tape, loss)
        
        terms.update(dict(
            q=q,
            returns=returns,
            loss=loss,
        ))

        return terms

    @tf.function
    def _sync_target_nets(self):
        [tv.assign(mv) for mv, tv in zip(
            self.q.trainable_variables, self.target_q.trainable_variables)]
