import logging
import re
import tensorflow as tf

from core.elements.loss import Loss as LossBase
from core.log import do_logging
from utility import rl_loss
from utility.tf_utils import reduce_mean, explained_variance, standard_normalization

logger = logging.getLogger(__name__)

def prefix_name(terms, name):
    if name is not None:
        new_terms = {}
        for k, v in terms.items():
            new_terms[f'{name}/{k}'] = v
        return new_terms
    return terms


class POLossImpl(LossBase):
    def _pg_loss(
        self, 
        tape, 
        act_dist,
        action, 
        advantage, 
        mu_prob, 
        mu=None, 
        mu_mean=None, 
        mu_std=None, 
        action_mask=None, 
        sample_mask=None, 
        n=None, 
        name=None,
    ):
        terms = {}
        self.log_for_debug(
            tape, 
            terms, 
            advantage=advantage, 
            adv_unit_std=tf.math.reduce_std(advantage, axis=-1), 
        )
        if action_mask is not None:
            n_avail_actions = tf.reduce_sum(
                tf.cast(action_mask, tf.float32), -1)
            self.log_for_debug(
                tape, 
                terms, 
                n_avail_actions=n_avail_actions
            )

        tf.debugging.assert_all_finite(advantage, 'Bad advantage')
        pi_prob = act_dist.prob(action)
        tf.debugging.assert_all_finite(pi_prob, 'Bad new_logprob')
        entropy = act_dist.entropy()
        tf.debugging.assert_all_finite(entropy, 'Bad entropy')
        ratio = pi_prob / mu_prob
        pi_logprob = tf.math.log(pi_prob)
        raw_pg_loss, pg_loss = rl_loss.pg_loss(
            pg_coef=self.config.pg_coef, 
            advantage=advantage, 
            logprob=pi_logprob, 
            mask=sample_mask, 
            n=n, 
        )
        raw_entropy_loss, entropy_loss = rl_loss.entropy_loss(
            entropy_coef=self.config.entropy_coef, 
            entropy=entropy, 
            mask=sample_mask, 
            n=n
        )
        self.log_for_debug(
            tape, 
            terms, 
            ratio=ratio,
            approx_kl=tf.math.log(mu_prob) - pi_logprob, 
            entropy=entropy,
            raw_pg_loss=raw_pg_loss, 
            pg_loss=pg_loss, 
            raw_entropy_loss=raw_entropy_loss, 
            entropy_loss=entropy_loss, 
        )
        loss = pg_loss + entropy_loss

        self.log_for_debug(
            tape, 
            terms, 
            loss=loss, 
        )
        terms = prefix_name(terms, name)

        return terms, loss


class ValueLossImpl(LossBase):
    def _value_loss(
        self, 
        tape, 
        value, 
        target, 
        sample_mask=None,
        n=None, 
        name=None, 
    ):
        value_loss_type = getattr(self.config, 'value_loss', 'mse')
        v_clip_frac = 0
        if value_loss_type == 'huber':
            raw_value_loss = rl_loss.huber_loss(
                value, 
                target, 
                threshold=self.config.huber_threshold
            )
        elif value_loss_type == 'mse':
            raw_value_loss = .5 * (value - target)**2
        else:
            raise ValueError(f'Unknown value loss type: {value_loss_type}')

        loss = reduce_mean(raw_value_loss, sample_mask, n)
        loss = self.config.value_coef * loss

        if self.config.get('debug', True):
            with tape.stop_recording():
                ev = explained_variance(target, value)
                terms = dict(
                    value=value,
                    raw_v_loss=raw_value_loss,
                    v_loss=loss,
                    explained_variance=ev,
                    v_target_std=tf.math.reduce_std(target, axis=-1), 
                    v_clip_frac=v_clip_frac,
                )
                terms = prefix_name(terms, name)
        else:
            terms = {}

        return terms, loss


class Loss(ValueLossImpl, POLossImpl):
    def loss(
        self, 
        obs, 
        action, 
        reward, 
        discount, 
        reset, 
        mu_prob, 
        mu=None, 
        mu_mean=None, 
        mu_std=None, 
        state=None, 
        action_mask=None, 
        life_mask=None, 
        mask=None
    ):
        sample_mask = life_mask
        n = None if sample_mask is None else tf.reduce_sum(sample_mask)
        with tf.GradientTape() as tape:
            x, _ = self.model.encode(
                x=obs, 
                state=state, 
                mask=mask
            )

            value = self.value(x)
            x = x[:, :-1]
            act_dist = self.policy(x)
            pi_prob = act_dist.prob(action)

            next_value = value[:, 1:]
            value = value[:, :-1]
            with tape.stop_recording():
                v_target, advantage = rl_loss.v_trace(
                    reward=reward, 
                    value=value, 
                    next_value=next_value, 
                    pi=pi_prob, 
                    mu=mu_prob, 
                    discount=discount, 
                    reset=reset, 
                    gamma=self.config.gamma, 
                    lambda_=self.config.lam, 
                    c_clip=self.config.c_clip, 
                    rho_clip=self.config.rho_clip, 
                    rho_clip_pg=self.config.rho_clip, 
                    axis=1
                )
            actor_terms, actor_loss = self._pg_loss(
                tape=tape, 
                act_dist=act_dist, 
                action=action, 
                advantage=advantage, 
                mu_prob=mu_prob, 
                mu=mu, 
                mu_mean=mu_mean, 
                mu_std=mu_std, 
                action_mask=action_mask,
                sample_mask=sample_mask,
                n=n
            )

            value = self.value(x)
            value_terms, value_loss = self._value_loss(
                tape=tape, 
                value=value,
                target=v_target, 
                sample_mask=sample_mask, 
                n=n, 
            )

            loss = actor_loss + value_loss
        
        terms = {**actor_terms, **value_terms}
        if self.config.get('debug', True) and life_mask is not None:
                terms['n_alive_units'] = tf.reduce_sum(
                    life_mask, -1)

        return tape, loss, terms

def create_loss(config, model, name='zero'):
    return Loss(config=config, model=model, name=name)
