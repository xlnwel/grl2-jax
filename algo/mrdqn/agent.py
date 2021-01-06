import numpy as np
import tensorflow as tf

from utility.rl_utils import *
from utility.rl_loss import retrace
from utility.schedule import TFPiecewiseSchedule
from core.tf_config import build
from core.optimizer import Optimizer
from algo.dqn.base import DQNBase

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
        mask=((batch_size, sample_size+1), dtype),
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

class Agent(DQNBase):
    def _construct_optimizers(self):
        if self._schedule_lr:
            self._lr = TFPiecewiseSchedule( [(5e5, self._lr), (2e6, 5e-5)])
        models = [self.encoder, self.rnn, self.q]
        self._optimizer = Optimizer(
            self._optimizer, models, self._lr, clip_norm=self._clip_norm)

    def _add_attributes(self):
        self._state = None
        self._prev_action = None
        self._prev_reward = None

    def _build_learn(self, env):
        # Explicitly instantiate tf.function to initialize variables
        TensorSpecs = dict(
            obs=((self._sample_size+1, *env.obs_shape), tf.float32, 'obs'),
            action=((self._sample_size+1,), tf.int32, 'prev_action'),
            reward=((self._sample_size+1,), tf.float32, 'prev_reward'),
            logpi=((self._sample_size,), tf.float32, 'logpi'),
            discount=((self._sample_size,), tf.float32, 'discount'),
            mask=((self._sample_size+1,), tf.float32, 'mask')
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

    def reset_states(self, state=None):
        if state is None:
            self._state, self._prev_action, self._prev_reward = None, None, None
        else:
            self._state, self._prev_action, self._prev_reward= state

    def get_states(self):
        return self._state, self._prev_action, self._prev_reward

    def __call__(self, obs, reset=np.zeros(1), evaluation=False, 
                env_output=None, **kwargs):
        eps = self._get_eps(evaluation)
        if obs.ndim % 2 != 0:
            obs = np.expand_dims(obs, 0)    # add batch dimension
        assert obs.ndim in (2, 4), obs.shape
        self._prev_reward = env_output.reward

        if self._state is None:
            self._state = self.q.get_initial_state(batch_size=tf.shape(obs)[0])
            if self.model.additional_rnn_input:
                self._prev_action = np.zeros(obs.shape[0], dtype=np.int32)
                self._prev_reward = np.zeros(obs.shape[0])
        mask = 1 - reset   # mask is applied in LSTM
        prev_state = self._state
        out, self._state = self.q.action(
                obs, self._state, True, 
                prev_action=self._prev_action,
                prev_reward=self._prev_reward)
        out = tf.nest.map_structure(lambda x: x.numpy(), out)
        if not evaluation:
            terms = out[1]
            if self._store_state:
                terms.update(prev_state._asdict())
            terms['mask'] = mask
            if self.model.additional_rnn_input:
                terms['prev_action'] = self._prev_action
        if self.model.additional_rnn_input:
            self._prev_action = out[0] if isinstance(out, tuple) else out
        return out

    @tf.function
    def _learn(self, obs, action, reward, discount, prob, mask, 
                IS_ratio=1, state=None, additional_rnn_input=[]):
        loss_fn = dict(
            huber=huber_loss, mse=lambda x: .5 * x**2)[self._loss_type]
        terms = {}
        if additional_rnn_input != []:
            prev_action, prev_reward = additional_rnn_input
            prev_action = tf.concat([prev_action, action[:, :-1]], axis=1)
            prev_reward = tf.concat([prev_reward, reward[:, :-1]], axis=1)
            add_inp = [prev_action, prev_reward]
        else:
            add_inp = additional_rnn_input
        mask = tf.expand_dims(mask, -1)
        with tf.GradientTape() as tape:
            embed = self.encoder(obs)
            t_embed = self.target_encoder(obs)
            if self._burn_in:
                bis = self._burn_in_size
                ss = self._sample_size - bis
                bi_embed, embed = tf.split(embed, [bis, ss+1], 1)
                tbi_embed, t_embed = tf.split(t_embed, [bis, ss+1], 1)
                if add_inp != []:
                    bi_add_inp, add_inp = zip(
                        *[tf.split(v, [bis, ss+1]) for v in add_inp])
                else:
                    bi_add_inp = []
                bi_mask, mask = tf.split(mask, [bis, ss+1], 1)
                bi_discount, discount = tf.split(discount, [bis, ss], 1)
                _, prob = tf.split(prob, [bis, ss], 1)
                _, o_state = self.rnn(bi_embed, state, bi_mask,
                    additional_input=bi_add_inp)
                _, t_state = self.target_rnn(tbi_embed, state, bi_mask,
                    additional_input=bi_add_inp)
                o_state = tf.nest.map_structure(tf.stop_gradient, o_state)
            else:
                o_state = t_state = state
                ss = self._sample_size

            x, _ = self.rnn(embed, o_state, mask,
                additional_input=add_inp)
            t_x, _ = self.target_rnn(t_embed, t_state, mask,
                additional_input=add_inp)
            
            curr_x = x[:, :-1]
            next_x = x[:, 1:]
            t_next_x = t_x[:, 1:]
            curr_action = action[:, :-1]
            next_action = action[:, 1:]
            discount = discount * self._gamma
            
            q = self.q(curr_x, curr_action)
            next_qs = self.q.action(next_x)
            new_next_action = tf.argmax(next_qs, axis=-1, output_type=tf.int32)
            next_pi = tf.one_hot(new_next_action, self._action_dim, dtype=tf.float32)
            t_next_qs = self.target_q(t_next_x)
            next_mu_a = prob[:, 1:]
            returns = retrace(
                reward, t_next_qs, next_action, 
                next_pi, next_mu_a, discount,
                lambda_=self._lambda, 
                axis=1, tbo=self._tbo)
            returns = tf.stop_gradient(returns)
            error = returns - q
            loss = tf.reduce_mean(IS_ratio[:, None] * loss_fn(error))
        tf.debugging.assert_shapes([
            [q, (None, ss)],
            [next_qs, (None, ss, self._action_dim)],
            [next_action, (None, ss-1)],
            [curr_action, (None, ss)√],
            [next_pi, (None, ss, self._action_dim)],
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
        mvars = [v.variables for v in [self.encoder, self.rnn, self.q]]
        tvars = [v.variables for v in [self.target_encoder, self.target_rnn, self.target_q]]
        [tv.assign(mv) for mv, tv in zip(mvars, tvars)]
