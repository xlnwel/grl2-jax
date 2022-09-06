import tensorflow as tf

from core.elements.loss import Loss, LossEnsemble
from jax_utils import jax_loss
from tools.tf_utils import reduce_mean, explained_variance, standard_normalization


def prefix_name(terms, name):
    if name is not None:
        new_terms = {}
        for k, v in terms.items():
            new_terms[f'{name}/{k}'] = v
        return new_terms
    return terms


class PGLossImpl(Loss):
    def _pg_loss(
        self, 
        tape, 
        act_dist,
        action, 
        advantage, 
        logprob, 
        action_mask=None, 
        sample_mask=None, 
        n=None, 
        name=None,
    ):
        with tape.stop_recording():
            raw_adv = advantage
            if self.config.norm_adv:
                advantage = standard_normalization(advantage, mask=sample_mask)

        new_logprob = act_dist.log_prob(action)
        tf.debugging.assert_all_finite(new_logprob, 'Bad new_logprob')
        raw_entropy = act_dist.entropy()
        tf.debugging.assert_all_finite(raw_entropy, 'Bad entropy')
        log_ratio = new_logprob - logprob
        ratio = tf.exp(log_ratio)
        loss_pg, loss_clip, raw_ppo_loss, ppo_loss, clip_frac = \
            loss.ppo_loss(
                pg_coef=self.config.pg_coef, 
                advantage=advantage, 
                ratio=ratio, 
                clip_range=self.config.ppo_clip_range, 
                mask=sample_mask, 
                n=n, 
            )
        raw_entropy_loss, entropy_loss = loss.entropy_loss(
            entropy_coef=self.config.entropy_coef, 
            entropy=raw_entropy, 
            mask=sample_mask, 
            n=n
        )
        loss = ppo_loss + entropy_loss

        if self.config.get('debug', True):
            with tape.stop_recording():
                terms = dict(
                    raw_adv=raw_adv, 
                    advantage=advantage, 
                    ratio=tf.exp(log_ratio),
                    raw_entropy=raw_entropy,
                    entropy=raw_entropy,
                    raw_entropy_loss=raw_entropy_loss,
                    approx_kl=.5 * reduce_mean((log_ratio)**2, sample_mask, n),
                    new_logprob=new_logprob, 
                    raw_pg_loss=loss_pg, 
                    raw_clipped_loss=loss_clip, 
                    p_clip_frac=clip_frac,
                    raw_ppo_loss=raw_ppo_loss,
                    ppo_loss=ppo_loss,
                    entropy_loss=entropy_loss, 
                    actor_loss=loss,
                    adv_std=tf.math.reduce_std(advantage, axis=-1), 
                )
                if not self.model.policy.is_action_discrete:
                    new_mean = act_dist.mean()
                    new_std = tf.exp(self.policy.logstd)
                    terms['pi_mean'] = new_mean
                    terms['pi_std'] = new_std
                if action_mask is not None:
                    terms['n_avail_actions'] = tf.reduce_sum(
                        tf.cast(action_mask, tf.float32), -1)
                terms = prefix_name(terms, name)
        else:
            terms = {}

        return terms, loss


class ValueLossImpl(Loss):
    def _value_loss(
        self, 
        tape, 
        value, 
        traj_ret, 
        old_value, 
        sample_mask=None,
        n=None, 
        name=None, 
    ):
        value_loss_type = getattr(self.config, 'value_loss', 'mse')
        v_clip_frac = 0
        if value_loss_type == 'huber':
            raw_value_loss = jax_loss.huber_loss(
                value, 
                traj_ret, 
                threshold=self.config.huber_threshold
            )
        elif value_loss_type == 'mse':
            raw_value_loss = .5 * (value - traj_ret)**2
        elif value_loss_type == 'clip' or value_loss_type == 'clip_huber':
            raw_value_loss, v_clip_frac = jax_loss.clipped_value_loss(
                value, 
                traj_ret, 
                old_value, 
                self.config.value_clip_range, 
                huber_threshold=self.config.get('huber_threshold', None), 
                mask=sample_mask, 
                n=n,
            )
        else:
            raise ValueError(f'Unknown value loss type: {value_loss_type}')
        
        raw_value_loss, value_loss = jax_loss.to_loss(
            raw_value_loss, 
            coef=self.config.value_coef, 
            mask=sample_mask, 
            n=n
        )

        if self.config.get('debug', True):
            with tape.stop_recording():
                ev = explained_variance(traj_ret, value)
                terms = dict(
                    value=value,
                    raw_v_loss=raw_value_loss,
                    v_loss=value_loss,
                    explained_variance=ev,
                    traj_ret_std=tf.math.reduce_std(traj_ret, axis=-1), 
                    v_clip_frac=v_clip_frac,
                )
                terms = prefix_name(terms, name)
        else:
            terms = {}

        return terms, value_loss


class PPOPolicyLoss(PGLossImpl):
    def loss(
        self, 
        obs, 
        action, 
        advantage, 
        logprob, 
        prev_reward=None, 
        prev_action=None, 
        state=None, 
        action_mask=None, 
        life_mask=None, 
        mask=None
    ):
        sample_mask = life_mask if self.config.life_mask else None
        n = None if sample_mask is None else tf.reduce_sum(sample_mask)
        with tf.GradientTape() as tape:
            x, _ = self.model.encode(
                x=obs, 
                prev_reward=prev_reward, 
                prev_action=prev_action, 
                state=state, 
                mask=mask
            )
            act_dist = self.policy(x, action_mask)
            terms, loss = self._pg_loss(
                tape=tape, 
                act_dist=act_dist, 
                action=action, 
                advantage=advantage, 
                logprob=logprob, 
                action_mask=action_mask, 
                sample_mask=sample_mask, 
                n=n
            )

        if life_mask is not None:
            terms['n_alive_units'] = tf.reduce_sum(
                life_mask, -1)

        return tape, loss, terms


class PPOValueLoss(ValueLossImpl):
    def loss(
        self, 
        global_state, 
        value, 
        traj_ret, 
        prev_reward=None, 
        prev_action=None, 
        state=None, 
        life_mask=None, 
        mask=None
    ):
        old_value = value
        sample_mask = life_mask if self.config.life_mask else None
        n = None if sample_mask is None else tf.reduce_sum(sample_mask)
        with tf.GradientTape() as tape:
            value, _ = self.model.compute_value(
                global_state=global_state,
                prev_reward=prev_reward,
                prev_action=prev_action,
                state=state,
                mask=mask
            )

            terms, loss = self._value_loss(
                tape=tape, 
                value=value, 
                traj_ret=traj_ret, 
                old_value=old_value, 
                sample_mask=sample_mask,
                n=n,
            )

        return tape, loss, terms


def create_loss(config, model, name='ppo'):
    def constructor(config, cls, name):
        return cls(
            config=config, 
            model=model[name], 
            name=name)

    return LossEnsemble(
        config=config,
        model=model,
        constructor=constructor,
        name=name,
        policy=PPOPolicyLoss,
        value=PPOValueLoss,
    )
