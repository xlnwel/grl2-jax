import tensorflow as tf

from utility.tf_utils import softmax, log_softmax
from utility.rl_utils import *
from utility.rl_loss import retrace
from algo.mrdqn.base import RDQNBase, get_data_format, collect


class Agent(RDQNBase):
    """ MRDQN methods """
    @tf.function
    def _learn(self, obs, action, reward, discount, mu, mask, 
                IS_ratio=1, state=None, additional_rnn_input=[]):
        mask = tf.expand_dims(mask, -1)
        if additional_rnn_input != []:
            prev_action, prev_reward = additional_rnn_input
            prev_action = tf.concat([prev_action, action[:, :-1]], axis=1)
            prev_reward = tf.concat([prev_reward, reward[:, :-1]], axis=1)
            add_inp = [prev_action, prev_reward]
        else:
            add_inp = additional_rnn_input
            
        target, terms = self._compute_target(
            obs, action, reward, discount, 
            mu, mask, state, add_inp)
        if self._burn_in:
            bis = self._burn_in_size
            ss = self._sample_size - bis
            bi_obs, obs, _ = tf.split(obs, [bis, ss, 1], 1)
            bi_mask, mask, _ = tf.split(mask, [bis, ss, 1], 1)
            if add_inp:
                bi_add_inp, add_inp, _ = zip(*[tf.split(v, [bis, ss, 1]) for v in add_inp])
            else:
                bi_add_inp = []
            _, state = self._compute_embed(bi_obs, bi_mask, state, bi_add_inp)
        else:
            obs, _ = tf.split(obs, [self._sample_size, 1], 1)
            mask, _ = tf.split(mask, [self._sample_size, 1], 1)
        action, _ = tf.split(action, [self._sample_size, 1], 1)

        with tf.GradientTape() as tape:
            x, _ = self._compute_embed(obs, mask, state, add_inp)
            
            q = self.q(x, action)
            error = target - q
            loss = tf.reduce_mean(.5 * error**2, axis=-1)
            loss = tf.reduce_mean(IS_ratio * loss)
        tf.debugging.assert_shapes([
            [q, (None, self._sample_size)],
            [target, (None, self._sample_size)],
            [error, (None, self._sample_size)],
            [IS_ratio, (None,)],
            [loss, ()]
        ])
        if self._is_per:
            priority = self._compute_priority(tf.abs(error))
            terms['priority'] = priority
        
        terms['norm'] = self._optimizer(tape, loss)
        
        terms.update(dict(
            q=q,
            mu_min=tf.reduce_min(mu),
            mu=mu,
            mu_std=tf.math.reduce_std(mu),
            target=target,
            loss=loss,
        ))

        return terms
    
    def _compute_target(self, obs, action, reward, discount, 
                        mu, mask, state, add_inp):
        terms = {}
        x, _ = self._compute_embed(obs, mask, state, add_inp, online=False)
        if self._burn_in:
            bis = self._burn_in_size
            ss = self._sample_size - bis
            _, reward = tf.split(reward, [bis, ss], 1)
            _, discount = tf.split(discount, [bis, ss], 1)
            _, next_mu_a = tf.split(mu, [bis+1, ss], 1)
            _, next_x = tf.split(x, [bis+1, ss], 1)
            _, next_action = tf.split(action, [bis+1, ss], 1)
        else:
            _, next_mu_a = tf.split(mu, [1, self._sample_size], 1)
            _, next_x = tf.split(x, [1, self._sample_size], 1)
            _, next_action = tf.split(action, [1, self._sample_size], 1)

        next_qs = self.target_q(next_x)
        regularization = None
        if self._probabilistic_regularization is None:
            if self._double:
                online_x, _ = self._compute_embed(obs, mask, state, add_inp)
                next_online_x = tf.split(online_x, [bis+1, ss-1], 1)
                next_online_qs = self.q(next_online_x)
                next_pi = self.compute_greedy_action(next_online_qs, one_hot=True)
            else:    
                next_pi = self.target_q.compute_greedy_action(next_qs, one_hot=True)
        elif self._probabilistic_regularization == 'mu':
            next_pi = softmax(next_qs, self._tau)
        elif self._probabilistic_regularization == 'entropy':
            next_pi = softmax(next_qs, self._tau)
            next_logpi = log_softmax(next_qs, self._tau)
            regularization = tf.reduce_sum(next_pi * next_logpi, axis=-1)
            terms['next_entropy'] = - regularization / self._tau
        else:
            raise ValueError(self._probabilistic_regularization)

        discount = discount * self._gamma
        target = retrace(
            reward, next_qs, next_action, 
            next_pi, next_mu_a, discount,
            lambda_=self._lambda, 
            axis=1, tbo=self._tbo,
            regularization=regularization)

        return target, terms
