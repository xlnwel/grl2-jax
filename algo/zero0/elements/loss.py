import logging
import tensorflow as tf

from core.elements.loss import LossEnsemble
from core.log import do_logging
from jax_utils import jax_loss
from tools.tf_utils import assert_shape_compatibility, explained_variance
from algo.zero.elements.loss import prefix_name, POLossImpl, ValueLossImpl

logger = logging.getLogger(__name__)


class Loss(ValueLossImpl, POLossImpl):
    def loss(
        self, 
        *, 
        tape, 
        obs, 
        next_obs=None, 
        action, 
        old_value, 
        reward, 
        discount, 
        reset, 
        mu_logprob, 
        mu=None, 
        mu_mean=None, 
        mu_std=None, 
        state=None, 
        action_mask=None, 
        life_mask=None, 
        mask=None, 
        name=None, 
        use_meta=None, 
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
        value = self.meta_value(x)
        if next_obs is None:
            x = x[:, :-1]
            next_value = value[:, 1:]
            value = value[:, :-1]
        else:
            with tape.stop_recording():
                assert state is None, 'unexpected states'
                next_x, _ = self.model.encode(next_obs)
                next_value = self.meta_value(next_x)
        act_dist = self.policy(x)
        pi_logprob = act_dist.log_prob(action)
        assert_shape_compatibility([pi_logprob, mu_logprob])
        log_ratio = pi_logprob - mu_logprob
        ratio = tf.exp(log_ratio)
        ratio = tf.stop_gradient(ratio)
        # tf.debugging.assert_near(
        #     tf.where(tf.cast(reset, bool), 0., log_ratio), 0., 1e-5, 1e-5)

        with tape.stop_recording():
            v_target, advantage = loss.v_trace_from_ratio(
                reward=reward, 
                value=value, 
                next_value=next_value, 
                ratio=ratio, 
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
            pi_logprob=pi_logprob, 
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
            lam=lam, 
            pi_logprob=pi_logprob, 
            ratio=ratio, 
            approx_kl=log_ratio, 
            loss=loss, 
        )
        terms.update(actor_terms)
        terms.update(value_terms)
        terms = prefix_name(terms, name)

        if debug:
            if mu is not None:
                terms['diff_pi'] = tf.nn.softmax(act_dist.logits) - mu
            elif mu_mean is not None:
                terms['pi_mean'] = act_dist.loc
                terms['diff_pi_mean'] = act_dist.loc - mu_mean
                pi_std = tf.exp(self.policy.logstd)
                terms['pi_std'] = pi_std
                terms['diff_pi_std'] = pi_std - mu_std
                # tf.debugging.assert_equal(pi_std, mu_std)
                # tf.debugging.assert_equal(act_dist.loc, mu_mean)

        return loss, terms

    def bmg_loss(
        self, 
        *, 
        tape, 
        obs, 
        next_obs=None, 
        action, 
        old_value, 
        reward, 
        discount, 
        reset, 
        mu_logprob, 
        mu=None, 
        mu_mean=None, 
        mu_std=None, 
        state=None, 
        action_mask=None, 
        sample_mask=None, 
        n=None, 
        mask=None, 
        name=None, 
        use_meta=None, 
        debug=True, 
    ):
        _, act_dist, value = self.model.forward(
            obs=obs[:, :-1] if next_obs is None else obs, 
            state=state, 
            mask=mask
        )
        logprob = act_dist.prob(action)
        sample_prob = tf.exp(mu_logprob)

        is_action_discrete = self.model.policy.is_action_discrete
        if is_action_discrete:
            pi = tf.nn.softmax(act_dist.logits)
            pi_mean = None
            pi_std = None
        else:
            pi = None
            pi_mean = act_dist.loc
            pi_std = tf.exp(self.model.policy.logstd)

        kl, raw_kl_loss, kl_loss = loss.compute_kl(
            kl_type=self.config.kl,
            kl_coef=self.config.kl_coef,
            logp=mu_logprob, 
            logq=logprob, 
            sample_prob=sample_prob, 
            pi1=mu,
            pi2=pi,
            pi1_mean=mu_mean,
            pi2_mean=pi_mean,
            pi1_std=mu_std,
            pi2_std=pi_std,
            sample_mask=sample_mask,
            n=n, 
            pi_mask=action_mask,
        )
        actor_loss = kl_loss

        value_loss, value_terms = self._value_loss(
            tape=tape, 
            value=value, 
            target=old_value, 
            sample_mask=sample_mask, 
            n=n, 
            name=name, 
            use_meta=use_meta, 
            debug=debug
        )

        loss = actor_loss + value_loss

        terms = value_terms
        self.log_for_debug(
            tape, 
            terms, 
            kl=kl, 
            raw_kl_loss=raw_kl_loss, 
            kl_loss=kl_loss, 
            actor_loss=actor_loss, 
            loss=loss, 
        )
        terms = prefix_name(terms, name)

        return loss, terms

def create_loss(config, model, name='zero'):
    rl_loss = Loss(config=config, model=model['rl'], name='rl')
    meta_loss = Loss(config=config, model=model['meta'], name='meta')

    return LossEnsemble(
        config=config, 
        model=model, 
        components=dict(
            rl=rl_loss, 
            meta=meta_loss
        ), 
        name=name, 
    )
