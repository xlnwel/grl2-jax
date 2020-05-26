import numpy as np
import tensorflow as tf

from utility.display import pwc
from utility.rl_utils import *
from utility.utils import Every
from utility.schedule import TFPiecewiseSchedule, PiecewiseSchedule
from utility.timer import TBTimer
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config, step_track
from core.optimizer import Optimizer


class Agent(BaseAgent):
    @agent_config
    def __init__(self, *, dataset, env):
        self._is_per = dataset.buffer_type().endswith('per')
        self.dataset = dataset

        if self._schedule_lr:
            self._lr = TFPiecewiseSchedule(
                [(5e5, self._lr), (2e6, 5e-5)], outside_value=5e-5)
        if self._schedule_eps:
            self._act_eps = PiecewiseSchedule(((5e4, 1), (4e5, .02)))

        self._optimizer = Optimizer(self._optimizer, self.q, self._lr, clip_norm=self._clip_norm)
        self._ckpt_models['optimizer'] = self._optimizer

        self._state = None

        self._obs_shape = env.obs_shape
        self._action_dim = env.action_dim

        # Explicitly instantiate tf.function to initialize variables
        TensorSpecs = dict(
            obs=((self._sample_size, *env.obs_shape), self._dtype, 'obs'),
            action=((self._sample_size, env.action_dim,), self._dtype, 'action'),
            reward=((self._sample_size,), self._dtype, 'reward'),
            logpi=((self._sample_size,), self._dtype, 'logpi'),
            discount=((self._sample_size,), self._dtype, 'discount'),
        )
        if self._is_per:
            TensorSpecs['IS_ratio'] = ((), self._dtype, 'IS_ratio')
        if self._store_state:
            state_size = self.q.state_size
            TensorSpecs['state'] = (
               [((sz, ), self._dtype, name) 
               for name, sz in zip(['h', 'c'], state_size)]
            )
        self.learn = build(self._learn, TensorSpecs)

        self._to_sync = Every(self._target_update_period)
        self._sync_target_nets()

    def reset_states(self, state=None):
        self._state = state

    def get_states(self):
        return self._state

    def __call__(self, obs, reset=np.zeros(1), deterministic=False):
        if self._schedule_eps:
            eps = self._act_eps.value(self.env_steps)
            self.store(act_eps=eps)
        else:
            eps = self._act_eps
        obs = np.reshape(obs, (-1, *self._obs_shape))
        if self._state is None:
            self._state = self.q.get_initial_state(batch_size=tf.shape(obs)[0])
        if np.any(reset):
            mask = tf.cast(1. - reset, self._dtype)
            self._state = tf.nest.map_structure(lambda x: x * mask, self._state)
        prev_state = self._state
        if deterministic:
            action, self._state = self.q.action(obs, self._state, deterministic)
            return action.numpy()
        else:
            action, terms, self._state = self.q.action(obs, self._state, deterministic, eps)
            if self._store_state:
                terms['h'] = prev_state[0]
                terms['c'] = prev_state[1]
            terms = tf.nest.map_structure(lambda x: np.squeeze(x.numpy()), terms)
            return action.numpy(), terms

    @step_track
    def learn_log(self, step):
        for i in range(self.N_UPDATES):
            with TBTimer('sample', 2500):
                data = self.dataset.sample()
            if self._is_per:
                idxes = data['idxes'].numpy()
                del data['idxes']
            with TBTimer('learn', 2500):
                terms = self.learn(**data)
            if self._to_sync(self.train_steps+i):
                self._sync_target_nets()

            if self._schedule_lr:
                terms['lr'] = self._lr(self._env_steps)
            terms = {k: v.numpy() for k, v in terms.items()}

            if self._is_per:
                self.dataset.update_priorities(terms['priority'], idxes)
            self.store(**terms)
        return self.N_UPDATES

    @tf.function
    def _learn(self, obs, action, reward, discount, logpi, state=None, IS_ratio=1):
        loss_fn = dict(
            huber=huber_loss, mse=lambda x: .5 * x**2)[self._loss_type]
        terms = {}
        with tf.GradientTape() as tape:
            embed = self.q.cnn(obs)
            t_embed = self.target_q.cnn(obs)
            if self._burn_in:
                bis = self._burn_in_size
                ss = self._sample_size - bis
                bi_embed, embed = tf.split(embed, [bis, ss], 1)
                tbi_embed, t_embed = tf.split(t_embed, [bis, ss], 1)
                bi_action, action = tf.split(action, [bis, ss], 1)
                bi_reward, reward = tf.split(reward, [bis, ss], 1)
                bi_discount, discount = tf.split(discount, [bis, ss], 1)
                _, logpi = tf.split(logpi, [bis, ss], 1)
                if self._add_input:
                    bi_rnn_input = tf.concat([bi_embed, bi_action], -1)
                    tbi_rnn_input = tf.concat([tbi_embed, bi_action], -1)
                else:
                    bi_rnn_input = bi_embed
                    tbi_rnn_input = tbi_embed
                _, o_state = self.q.rnn(bi_rnn_input, state)
                _, t_state = self.target_q.rnn(tbi_rnn_input, state)
                o_state = tf.nest.map_structure(lambda x: tf.stop_gradient(x), o_state)
            else:
                o_state = t_state = state
                ss = self._sample_size
            if self._add_input:
                rnn_input = tf.concat([embed, action], -1)
                t_rnn_input = tf.concat([t_embed, action], -1)
            else:
                rnn_input = embed
                t_rnn_input = t_embed
            x, _ = self.q.rnn(rnn_input, o_state)
            t_x, _ = self.target_q.rnn(t_rnn_input, t_state)
            
            curr_x = x[:, :-1]
            next_x = x[:, 1:]
            t_next_x = t_x[:, 1:]
            curr_action = action[:, :-1]
            discount = discount[:, :-1] * self._gamma
            
            q = self.q.mlp(curr_x, curr_action)
            next_qs = self.q.mlp(next_x)
            next_action = tf.argmax(next_qs, axis=-1)
            t_next_q = self.target_q.mlp(t_next_x, next_action)
            next_prob = next_action == tf.argmax(action[:, 1:], axis=-1)
            next_prob = tf.cast(next_prob, logpi.dtype)
            ratio = next_prob / tf.math.exp(logpi[:, 1:])
            returns = retrace_lambda(
                reward[:, :-1], q, t_next_q, 
                ratio, discount, lambda_=self._lambda, 
                axis=1, tbo=self._tbo)
            returns = tf.stop_gradient(returns)
            error = returns - q
            loss = tf.reduce_mean(IS_ratio[:, None] * loss_fn(error))

        if self._is_per:
            priority = self._compute_priority(tf.abs(error))
            terms['priority'] = priority
        
        terms['norm'] = self._optimizer(tape, loss)
        
        terms.update(dict(
            q=q,
            logpi=logpi,
            ratio=ratio,
            returns=returns,
            loss=loss,
        ))

        return terms

    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        priority = (self._per_eta*tf.math.reduce_max(priority, axis=1) 
                    + (1-self._per_eta)*tf.math.reduce_mean(priority, axis=1))
        priority += self._per_epsilon
        priority **= self._per_alpha
        return priority

    @tf.function
    def _sync_target_nets(self):
        [tv.assign(mv) for mv, tv in zip(
            self.q.trainable_variables, self.target_q.trainable_variables)]
