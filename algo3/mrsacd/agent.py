import functools
import tensorflow as tf

from utility.rl_utils import *
from utility.rl_loss import retrace
from utility.tf_utils import explained_variance
from utility.schedule import TFPiecewiseSchedule
from core.optimizer import Optimizer
from core.decorator import override
from algo.mrdqn.agent import Agent as RAgent, get_data_format, collect


class Agent(RAgent):
    """ Initialization """
    @override(RAgent)
    def _construct_optimizers(self):
        if self._schedule_lr:
            assert isinstance(self._actor_lr, list), self._actor_lr
            assert isinstance(self._value_lr, list), self._value_lr
            self._actor_lr = TFPiecewiseSchedule(self._actor_lr)
            self._value_lr = TFPiecewiseSchedule(self._value_lr)
        
        PartialOpt = functools.partial(
            Optimizer,
            name=self._optimizer,
            weight_decay=getattr(self, '_weight_decay', None),
            clip_norm=getattr(self, '_clip_norm', None),
            epsilon=getattr(self, '_epsilon', 1e-7)
        )
        self._actor_opt = PartialOpt(models=self.actor, lr=self._actor_lr)
        value_models = [self.encoder, self.q]
        self._value_opt = PartialOpt(models=value_models, lr=self._value_lr)

        if self.temperature.is_trainable():
            self._temp_opt = Optimizer(self._optimizer, self.temperature, self._temp_lr)

    @tf.function
    def _learn(self, obs, action, reward, discount, prob, mask, 
                IS_ratio=1, state=None, additional_rnn_input=[]):
        if self.temperature.type == 'schedule':
            _, temp = self.temperature(self._train_step)
        elif self.temperature.type == 'state':
            raise NotImplementedError
        else:
            _, temp = self.temperature()

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
            x = self.encoder(obs)
            t_x = self.target_encoder(obs)
            ss = self._sample_size
            if 'rnn' in self.model:
                if self._burn_in:
                    bis = self._burn_in_size
                    ss = self._sample_size - bis
                    bi_x, x = tf.split(x, [bis, ss+1], 1)
                    tbi_x, t_x = tf.split(t_x, [bis, ss+1], 1)
                    if add_inp != []:
                        bi_add_inp, add_inp = zip(
                            *[tf.split(v, [bis, ss+1]) for v in add_inp])
                    else:
                        bi_add_inp = []
                    bi_mask, mask = tf.split(mask, [bis, ss+1], 1)
                    _, discount = tf.split(discount, [bis, ss], 1)
                    _, prob = tf.split(prob, [bis, ss], 1)
                    _, o_state = self.rnn(bi_x, state, bi_mask,
                        additional_input=bi_add_inp)
                    _, t_state = self.target_rnn(tbi_x, state, bi_mask,
                        additional_input=bi_add_inp)
                    o_state = tf.nest.map_structure(tf.stop_gradient, o_state)
                else:
                    o_state = t_state = state

                x, _ = self.rnn(x, o_state, mask,
                    additional_input=add_inp)
                t_x, _ = self.target_rnn(t_x, t_state, mask,
                    additional_input=add_inp)
            
            curr_x = x[:, :-1]
            next_x = x[:, 1:]
            t_next_x = t_x[:, 1:]
            curr_action = action[:, :-1]
            next_action = action[:, 1:]
            discount = discount * self._gamma
            
            qs = self.q(curr_x)
            q = tf.reduce_sum(qs * curr_action, axis=-1)
            next_pi, next_logps = self.target_actor.train_step(next_x)
            next_qs = self.target_q(t_next_x)
            if self._soft_target:
                next_qs = next_qs - temp * next_logps
            next_mu_a = prob[:, 1:]
            target = retrace(
                reward, next_qs, next_action, 
                next_pi, next_mu_a, discount,
                lambda_=self._lambda, 
                axis=1) # NOTE: tbo should be performed at next_qs before entropies are added
            target = tf.stop_gradient(target)
            error = target - q
            value_loss = tf.reduce_mean(.5 * error**2, axis=-1)
            value_loss = tf.reduce_mean(IS_ratio * value_loss)
        terms['value_norm'] = self._value_opt(tape, value_loss)
        tf.debugging.assert_shapes([
            [q, (None, ss)],
            [next_pi, (None, ss, self._action_dim)],
            [target, (None, ss)],
            [error, (None, ss)],
            [IS_ratio, (None,)],
            [value_loss, ()]
        ])

        with tf.GradientTape() as tape:
            act_probs, act_logps = self.actor.train_step(curr_x)
            q = tf.reduce_sum(act_probs * qs, axis=-1)
            entropy = - tf.reduce_sum(act_probs * act_logps, axis=-1)
            actor_loss = -tf.reduce_mean((q + temp * entropy), axis=-1)
            tf.debugging.assert_shapes([[actor_loss, (None,)]])
            actor_loss = tf.reduce_mean(IS_ratio * actor_loss)
        terms['actor_norm'] = self._actor_opt(tape, actor_loss)

        if self._is_per:
            # we intend to use error as priority instead of TD error used in the paper
            priority = self._compute_priority(tf.abs(error))
            terms['priority'] = priority
        
        terms.update(dict(
            q=q,
            prob=prob,
            target=target,
            value_loss=value_loss,
            actor_loss=actor_loss,
            explained_variance_q=explained_variance(target, q),
        ))

        return terms

    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        priority = (self._per_eta*tf.math.reduce_max(priority, axis=1) 
                    + (1-self._per_eta)*tf.math.reduce_mean(priority, axis=1))
        priority += self._per_epsilon
        priority **= self._per_alpha
        return priority

    def reset_states(self, state=None):
        if state is None:
            self._state, self._prev_action, self._prev_reward = None, None, None
        else:
            self._state, self._prev_action, self._prev_reward= state

    def get_states(self):
        return self._state, self._prev_action, self._prev_reward
