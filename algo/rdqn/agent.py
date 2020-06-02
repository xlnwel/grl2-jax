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


def get_data_format(env, batch_size, sample_size=None, 
        is_per=False, store_state=False, state_size=None, 
        dtype=tf.float32, **kwargs):
    obs_dtype = env.obs_dtype if len(env.obs_shape) == 3 else dtype
    data_format = dict(
        obs=((batch_size, sample_size+1, *env.obs_shape), obs_dtype),
        prev_action=((batch_size, sample_size+1, *env.action_shape), tf.int32),
        prev_reward=((batch_size, sample_size+1), dtype), 
        logpi=((batch_size, sample_size), dtype),
        discount=((batch_size, sample_size), dtype),
    )
    if is_per:
        data_format['IS_ratio'] = ((batch_size), dtype)
        data_format['idxes'] = ((batch_size), tf.int32)
    if store_state:
        from tensorflow.keras.mixed_precision.experimental import global_policy
        state_dtype = global_policy().compute_dtype
        data_format.update({
            k: ((batch_size, v), state_dtype)
                for k, v in state_size._asdict().items()
        })

    return data_format

class Agent(BaseAgent):
    @agent_config
    def __init__(self, *, dataset, env):
        self._is_per = dataset.name().endswith('per')
        self.dataset = dataset

        if self._schedule_lr:
            self._lr = TFPiecewiseSchedule(
                [(5e5, self._lr), (2e6, 5e-5)], outside_value=5e-5)

        self._optimizer = Optimizer(self._optimizer, self.q, self._lr, 
            clip_norm=self._clip_norm, epsilon=self._epsilon)

        self._state = None
        self._prev_action = None
        self._prev_reward = None

        self._obs_shape = env.obs_shape
        self._action_dim = env.action_dim

        # Explicitly instantiate tf.function to initialize variables
        TensorSpecs = dict(
            obs=((self._sample_size+1, *env.obs_shape), tf.float32, 'obs'),
            prev_action=((self._sample_size+1,), env.action_dtype, 'prev_action'),
            prev_reward=((self._sample_size+1,), tf.float32, 'prev_reward'),
            logpi=((self._sample_size,), tf.float32, 'logpi'),
            discount=((self._sample_size,), tf.float32, 'discount'),
        )
        if self._is_per:
            TensorSpecs['IS_ratio'] = ((), tf.float32, 'IS_ratio')
        if self._store_state:
            state_size = self.q.state_size
            TensorSpecs['state'] = (
               [((sz, ), self._dtype, name) 
               for name, sz in state_size._asdict().items()]
            )
        self.learn = build(self._learn, TensorSpecs, print_terminal_info=True)

        self._to_sync = Every(self._target_update_period)
        self._sync_target_nets()

    def reset_states(self, state=None):
        if state is None:
            self._state, self._prev_action, self._prev_reward = None, None, None
        else:
            self._state, self._prev_action, self._prev_reward= state

    def get_states(self):
        return self._state, self._prev_action, self._prev_reward

    def __call__(self, obs, reset=np.zeros(1), deterministic=False, env_output=0, **kwargs):
        if self._add_input:
            self._prev_reward = env_output.reward
        eps = self._act_eps
        obs = np.reshape(obs, (-1, *self._obs_shape))
        if self._state is None:
            self._state = self.q.get_initial_state(batch_size=tf.shape(obs)[0])
            if self._add_input:
                self._prev_action = tf.zeros(tf.shape(obs)[0])
                self._prev_reward = tf.zeros(tf.shape(obs)[0])
        if np.any(reset):
            mask = tf.cast(1. - reset, tf.float32)
            self._state = tf.nest.map_structure(lambda x: x * mask, self._state)
            if self._add_input:
                self._prev_action = self._prev_action * mask
                self._prev_reward = self._prev_reward * mask
        if deterministic:
            action, self._state = self.q.action(
                obs, self._state, deterministic, 
                prev_action=self._prev_action,
                prev_reward=self._prev_reward)
            if self._add_input:
                self._prev_action = action
            return action.numpy()
        else:
            action, terms, state = self.q.action(
                obs, self._state, deterministic, eps,
                prev_action=self._prev_action,
                prev_reward=self._prev_reward)
            if self._store_state:
                terms['h'] = self._state[0]
                terms['c'] = self._state[1]
            terms = tf.nest.map_structure(lambda x: np.squeeze(x.numpy()), terms)
            self._state = state
            if self._add_input:
                self._prev_action = action
            return action.numpy(), terms

    @step_track
    def learn_log(self, step):
        for i in range(self.N_UPDATES):
            with TBTimer('sample', 1000):
                data = self.dataset.sample()
            if self._is_per:
                idxes = data['idxes'].numpy()
                del data['idxes']
            with TBTimer('learn', 1000):
                terms = self.learn(**data)
            if self._to_sync(self.train_step+i):
                self._sync_target_nets()

            if self._schedule_lr:
                terms['lr'] = self._lr(self._env_step)
            terms = {k: v.numpy() for k, v in terms.items()}

            if self._is_per:
                self.dataset.update_priorities(terms['priority'], idxes)
            self.store(**terms)
        return self.N_UPDATES

    @tf.function
    def _learn(self, obs, prev_action, prev_reward, discount, logpi, state=None, IS_ratio=1):
        loss_fn = dict(
            huber=huber_loss, mse=lambda x: .5 * x**2)[self._loss_type]
        terms = {}
        with tf.GradientTape() as tape:
            embed = self.q.cnn(obs)
            t_embed = self.target_q.cnn(obs)
            if self._burn_in:
                bis = self._burn_in_size
                ss = self._sample_size - bis
                bi_embed, embed = tf.split(embed, [bis, ss+1], 1)
                tbi_embed, t_embed = tf.split(t_embed, [bis, ss+1], 1)
                bi_prev_action, prev_action = tf.split(prev_action, [bis, ss+1], 1)
                bi_prev_reward, prev_reward = tf.split(prev_reward, [bis, ss+1], 1)
                bi_discount, discount = tf.split(discount, [bis, ss], 1)
                _, logpi = tf.split(logpi, [bis, ss], 1)
                if self._add_input:
                    pa, pr = bi_prev_action, bi_prev_reward
                else:
                    pa, pr = None, None
                _, o_state = self.q.rnn(bi_embed, state, 
                    prev_action=pa, prev_reward=pr)
                _, t_state = self.target_q.rnn(tbi_embed, state, 
                    prev_action=pa, prev_reward=pr)
                o_state = tf.nest.map_structure(tf.stop_gradient, o_state)
            else:
                o_state = t_state = state
                ss = self._sample_size
            if self._add_input:
                pa, pr = prev_action, prev_reward
            else:
                pa, pr = None, None
            x, _ = self.q.rnn(embed, o_state, 
                prev_action=pa, prev_reward=pr)
            t_x, _ = self.target_q.rnn(t_embed, t_state, 
                prev_action=pa, prev_reward=pr)
            
            curr_x = x[:, :-1]
            next_x = x[:, 1:]
            t_next_x = t_x[:, 1:]
            curr_action = prev_action[:, 1:]
            next_action = prev_action[:, 2:]
            reward = prev_reward[:, 1:]
            discount = discount * self._gamma
            
            q = self.q.mlp(curr_x, curr_action)
            next_qs = self.q.mlp(next_x)
            new_next_action = tf.argmax(next_qs, axis=-1, output_type=tf.int32)
            t_next_q = self.target_q.mlp(t_next_x, new_next_action)
            new_next_prob = tf.math.equal(new_next_action[:, :-1], next_action)
            new_next_prob = tf.cast(new_next_prob, logpi.dtype)
            next_ratio = new_next_prob / tf.math.exp(logpi[:, 1:])
            returns = retrace_lambda(
                reward, q, t_next_q, 
                next_ratio, discount, 
                lambda_=self._lambda, 
                axis=1, tbo=self._tbo)
            returns = tf.stop_gradient(returns)
            error = returns - q
            loss = tf.reduce_mean(IS_ratio[:, None] * loss_fn(error))
        tf.debugging.assert_shapes([
            [q, (None, ss)],
            [next_qs, (None, ss, self._action_dim)],
            [next_action, (None, ss-1)],
            [curr_action, (None, ss)],
            [new_next_prob, (None, ss-1)],
            [next_ratio, (None, ss-1)],
            [returns, (None, ss)],
            [error, (None, ss)],
            [IS_ratio, (None,)],
            [loss, ()]
        ])
        if self._is_per:
            # we intend to use error as priority instead of TD error used in the paper
            priority = self._compute_priority(tf.abs(error))
            terms['priority'] = priority
        
        terms['norm'] = self._optimizer(tape, loss)
        
        terms.update(dict(
            q=q,
            logpi=logpi,
            max_ratio=tf.math.reduce_max(next_ratio),
            ratio=next_ratio,
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
