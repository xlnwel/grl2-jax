import logging
import re
import tensorflow as tf

from core.elements.loss import Loss as LossBase
from core.log import do_logging
from utility import rl_loss
from utility.tf_utils import explained_variance, standard_normalization

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
        use_meta=False, 
        debug=True
    ):
        terms = {}
        self.log_for_debug(
            tape, 
            terms, 
            debug=debug, 
            advantage=advantage, 
            adv_unit_std=tf.math.reduce_std(advantage, axis=-1), 
        )
        if action_mask is not None:
            n_avail_actions = tf.reduce_sum(
                tf.cast(action_mask, tf.float32), -1)
            self.log_for_debug(
                tape, 
                terms, 
                debug=debug, 
                n_avail_actions=n_avail_actions
            )
        if sample_mask is not None:
            n_alive_units = tf.reduce_sum(sample_mask, -1)
            self.log_for_debug(
                tape, 
                terms, 
                debug=debug, 
                n_alive_units=n_alive_units
            )

        pg_coef = self.model.meta('pg_coef', inner=use_meta)
        entropy_coef = self.model.meta('entropy_coef', inner=use_meta)

        pi_prob = act_dist.prob(action)
        tf.debugging.assert_all_finite(pi_prob, 'Bad pi_prob')
        entropy = act_dist.entropy()
        tf.debugging.assert_all_finite(entropy, 'Bad entropy')
        ratio = pi_prob / mu_prob
        pi_logprob = tf.math.log(pi_prob)
        raw_pg_loss, pg_loss = rl_loss.pg_loss(
            pg_coef=pg_coef, 
            advantage=advantage, 
            logprob=pi_logprob, 
            mask=sample_mask, 
            n=n, 
        )
        raw_entropy_loss, entropy_loss = rl_loss.entropy_loss(
            entropy_coef=entropy_coef, 
            entropy=entropy, 
            mask=sample_mask, 
            n=n
        )
        self.log_for_debug(
            tape, 
            terms, 
            debug=debug, 
            pg_coef=pg_coef, 
            entropy_coef=entropy_coef, 
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
            debug=debug, 
            loss=loss, 
        )
        terms = prefix_name(terms, name)

        return loss, terms


class ValueLossImpl(LossBase):
    def _value_loss(
        self, 
        tape, 
        value, 
        target, 
        sample_mask=None,
        n=None, 
        name=None, 
        use_meta=False, 
        debug=False
    ):
        value_loss_type = getattr(self.config, 'value_loss', 'mse')
        value_coef = self.model.meta('value_coef', inner=use_meta)
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
        raw_value_loss, loss = rl_loss.to_loss(
            raw_value_loss, 
            coef=value_coef, 
            mask=sample_mask, 
            n=n
        )

        if debug and self.config.get('debug', True):
            with tape.stop_recording():
                ev = explained_variance(target, value)
                terms = dict(
                    value_coef=value_coef, 
                    value=value,
                    raw_v_loss=raw_value_loss,
                    v_loss=loss,
                    explained_variance=ev,
                    v_target_std=tf.math.reduce_std(target, axis=-1), 
                )
                terms = prefix_name(terms, name)
        else:
            terms = {}

        return loss, terms


class Loss(ValueLossImpl, POLossImpl):
    def loss(
        self, 
        tape, 
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
        mask=None, 
        name=None, 
        use_meta=False, 
        debug=True, 
    ):
        sample_mask = life_mask
        n = None if sample_mask is None else tf.reduce_sum(sample_mask)
        gamma = self.model.meta('gamma', inner=use_meta)
        lam = self.model.meta('lam', inner=use_meta)

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
                gamma=gamma, 
                lambda_=lam, 
                c_clip=self.config.c_clip, 
                rho_clip=self.config.rho_clip, 
                rho_clip_pg=self.config.rho_clip, 
                axis=1, 
            )
        actor_loss, actor_terms = self._pg_loss(
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
            n=n, 
            name=name, 
            use_meta=use_meta, 
            debug=debug
        )

        value_loss, value_terms = self._value_loss(
            tape=tape, 
            value=value,
            target=v_target, 
            sample_mask=sample_mask, 
            n=n, 
            name=name, 
            use_meta=use_meta, 
            debug=debug
        )

        loss = actor_loss + value_loss
        
        terms = {}
        self.log_for_debug(
            tape, 
            terms, 
            debug=debug, 
            gamma=gamma,
            lam=lam
        )
        prefix_name(terms, name)
        terms = {**actor_terms, **value_terms}

        return loss, terms

def create_loss(config, model, name='zero'):
    return Loss(config=config, model=model, name=name)
