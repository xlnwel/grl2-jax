from jax import lax, nn, random
import jax.numpy as jnp

from core.elements.loss import LossBase
from core.typing import AttrDict, dict2AttrDict
from jax_tools import jax_loss, jax_math
from jax_tools.jax_utils import split_data
from tools.utils import prefix_name


class Loss(LossBase):
    def post_init(self):
        if self.config.target_type == 'vtrace':
            assert self.config.pg_type != 'ppo', f'Repeatedly applying importance sampling.'

    def loss(
        self, 
        theta, 
        eta, 
        rng, 
        data, 
        name='theta', 
        use_meta=False, 
        use_dice=False, 
    ):
        rngs = random.split(rng, 2)
        stats = self.modules.meta_params(
            eta.meta_params if use_meta else {}, rngs[0], 
            inner=use_meta, stats=data.meta_param_stats
        )

        if data.action_mask is not None:
            stats.n_avail_actions = jnp.sum(data.action_mask, -1)
        if data.sample_mask is not None:
            stats.n_alive_units = jnp.sum(data.sample_mask, -1)

        act_dist, new_stats = self.model.forward(theta, rngs[1], data)
        stats.update(new_stats)
        stats = record_policy_stats(data, stats, act_dist)

        stats.v_target, stats.raw_adv = jax_loss.compute_target_advantage(
            config=self.config, 
            reward=data.reward, 
            discount=data.discount, 
            reset=data.reset, 
            value=lax.stop_gradient(stats.value), 
            next_value=stats.next_value, 
            ratio=lax.stop_gradient(stats.ratio), 
            gamma=stats.gamma, 
            lam=stats.lam, 
            axis=1
        )
        if self.config.stop_target_grads:
            stats.v_target = lax.stop_gradient(stats.v_target)

        stats = record_target_adv(stats)
        if self.config.norm_adv:
            stats.advantage = jax_math.standard_normalization(
                stats.raw_adv, 
                zero_center=self.config.get('zero_center', True), 
                mask=data.sample_mask, 
                n=data.n, 
                epsilon=self.config.get('epsilon', 1e-8), 
            )
        else:
            stats.advantage = stats.raw_adv
        stats.advantage = jax_math.clip(
            stats.advantage, self.config.adv_clip)

        actor_loss, stats = compute_actor_loss(
            self.config, data, stats, 
            act_dist=act_dist, use_dice=use_dice
        )
        value_loss, stats = compute_vf_loss(
            self.config, data, stats, 
        )
        loss = actor_loss + value_loss
        stats.loss = loss

        stats = prefix_name(stats, name)
        
        return loss, stats
    
    def eta_loss(
        self, 
        theta, 
        target_theta, 
        rng, 
        data, 
        name='eta', 
    ):
        if self.config.meta_type == 'plain':
            return self.loss(theta, {}, rng, data, name=name)
        elif self.config.meta_type == 'bmg':
            return self.bmg_loss(theta, target_theta, rng, data, name=name)
        else:
            raise NotImplementedError

    def bmg_loss(
        self, 
        theta, 
        target_theta, 
        rng, 
        data, 
        name='eta'
    ):
        rngs = random.split(rng, 3)
        stats = self.modules.meta_params(
            {}, rngs[0], inner=False
        )
        
        act_dist, new_stats = self.model.forward(
            theta, rngs[1], data)
        target_dist, target_stats = self.model.forward(
            target_theta, rngs[2], data)
        target_dist.stop_gradient()
        stats.update(new_stats)
        stats.v_target = lax.stop_gradient(target_stats.value)
        stats.entropy = act_dist.entropy()
        stats.target_logprob = target_dist.log_prob(data.action)
        stats[f'{name}/target/entropy'] = target_dist.entropy()
        for k, v in target_stats.items():
            stats[f'{name}/target/{k}'] = v
        stats = record_policy_stats(data, stats, act_dist)

        actor_loss, stats = compute_bmg_actor_loss(
            self.config, data, stats, act_dist, target_dist
        )
        value_loss, stats = compute_vf_loss(
            self.config, data, stats, 
        )

        loss = actor_loss + value_loss
        stats.loss = loss

        stats = prefix_name(stats, name)
        
        return loss, stats


def create_loss(config, model, name='zero'):
    loss = Loss(config=config, model=model, name=name)

    return loss


def compute_bmg_actor_loss(
    config, 
    data, 
    stats, 
    act_dist, 
    target_dist, 
):
    sample_mask, _ = split_data(
        data.sample_mask, data.next_sample_mask)
    if config.get('policy_life_mask', True):
        sample_mask = None

    p_stats = act_dist.get_stats('p')
    q_stats = target_dist.get_stats('q')
    stats.kl, stats.raw_actor_loss, stats.actor_loss = \
        jax_loss.compute_kl(
            kl_type=config.kl, 
            kl_coef=config.kl_coef, 
            logp=stats.pi_logprob, 
            logq=stats.target_logprob, 
            # sample_prob=data.mu_logprob, 
            **p_stats, 
            **q_stats, 
            sample_mask=sample_mask
        )

    return stats.actor_loss, stats

def compute_actor_loss(
    config, 
    data, 
    stats, 
    act_dist, 
    use_dice=False
):
    use_dice = config.use_dice and use_dice
    sample_mask, _ = split_data(
        data.sample_mask, data.next_sample_mask)
    if config.get('policy_life_mask', True):
        sample_mask = None
    # stats.pi_logprob = jnp.where(
    #     stats.pi_logprob > -1e-5, 0, stats.pi_logprob)
    # stats.ratio = jnp.where(
    #     stats.pi_logprob > -1e-5, 1, stats.ratio)

    if use_dice:
        dice_op = jax_loss.dice(
            stats.pi_logprob, 
            axis=config.dice_axis, 
            lam=config.dice_lam
        )
    else:
        dice_op = stats.pi_logprob
    stats.dice_op = dice_op

    if config.pg_type == 'pg':
        raw_pg_loss = jax_loss.pg_loss(
            advantage=stats.advantage, 
            logprob=dice_op, 
        )
    elif config.pg_type == 'ppo':
        if use_dice:
            ppo_pg_loss, ppo_clip_loss, raw_pg_loss = \
                jax_loss.high_order_ppo_loss(
                    advantage=stats.advantage, 
                    ratio=stats.ratio, 
                    dice_op=dice_op, 
                    clip_range=config.ppo_clip_range, 
                )
        else:
            ppo_pg_loss, ppo_clip_loss, raw_pg_loss = \
                jax_loss.ppo_loss(
                    advantage=stats.advantage, 
                    ratio=stats.ratio, 
                    clip_range=config.ppo_clip_range, 
                )
        stats.ppo_pg_loss = ppo_pg_loss
        stats.ppo_clip_loss = ppo_clip_loss
    else:
        raise NotImplementedError
    scaled_pg_loss, pg_loss = jax_loss.to_loss(
        raw_pg_loss, 
        stats.pg_coef, 
        mask=sample_mask, 
    )
    stats.raw_pg_loss = raw_pg_loss
    stats.scaled_pg_loss = scaled_pg_loss
    stats.pg_loss = pg_loss

    entropy = act_dist.entropy()
    scaled_entropy_loss, entropy_loss = jax_loss.entropy_loss(
        entropy_coef=stats.entropy_coef, 
        entropy=entropy, 
        mask=sample_mask, 
    )
    stats.entropy = entropy
    stats.scaled_entropy_loss = scaled_entropy_loss
    stats.entropy_loss = entropy_loss

    loss = pg_loss + entropy_loss
    stats.actor_loss = loss

    clip_frac = jax_math.mask_mean(
        lax.abs(stats.ratio - 1.) > config.get('ppo_clip_range', .2), 
        sample_mask, data.n)
    stats.clip_frac = clip_frac

    return loss, stats

def compute_vf_loss(
    config, 
    data, 
    stats, 
    new_stats=None, 
):
    if new_stats is None:
        new_stats = stats
    sample_mask, _ = split_data(
        data.sample_mask, data.next_sample_mask)
    if config.get('value_life_mask', False):
        sample_mask = data.sample_mask
    else:
        sample_mask = None

    value_loss_type = config.value_loss
    if value_loss_type == 'huber':
        raw_value_loss = jax_loss.huber_loss(
            stats.value, 
            y=stats.v_target, 
            threshold=config.huber_threshold
        )
    elif value_loss_type == 'mse':
        raw_value_loss = .5 * (stats.value - stats.v_target)**2
    elif value_loss_type == 'clip' or value_loss_type == 'clip_huber':
        raw_value_loss, new_stats.v_clip_frac = jax_loss.clipped_value_loss(
            stats.value, 
            stats.v_target, 
            data.old_value, 
            config.value_clip_range, 
            huber_threshold=config.huber_threshold, 
            mask=sample_mask, 
        )
    else:
        raise ValueError(f'Unknown value loss type: {value_loss_type}')
    new_stats.raw_v_loss = raw_value_loss
    scaled_value_loss, value_loss = jax_loss.to_loss(
        raw_value_loss, 
        coef=stats.value_coef, 
        mask=sample_mask, 
    )
    
    new_stats.scaled_v_loss = scaled_value_loss
    new_stats.v_loss = value_loss

    return value_loss, new_stats

def record_policy_stats(data, stats, act_dist):
    stats.diff_frac = jax_math.mask_mean(
        lax.abs(stats.pi_logprob - data.mu_logprob) > 1e-5, 
        data.sample_mask, data.n)
    stats.approx_kl = .5 * jax_math.mask_mean(
        (stats.log_ratio)**2, data.sample_mask, data.n)
    stats.approx_kl_max = jnp.max(.5 * (stats.log_ratio)**2)
    if data.mu_logits is not None:
        mu = nn.softmax(data.mu_logits)
        stats.pi = nn.softmax(act_dist.logits)
        stats.diff_pi = stats.pi - mu
    elif data.mu_mean is not None:
        stats.pi_mean = act_dist.mu
        stats.diff_pi_mean = act_dist.mu - data.mu_mean
        stats.pi_std = act_dist.std
        stats.diff_pi_std = act_dist.std - data.mu_std

    return stats

def record_target_adv(stats):
    stats.explained_variance = jax_math.explained_variance(
        stats.v_target, stats.value)
    stats.v_target_unit_std = jnp.std(stats.v_target, axis=-1)
    stats.raw_adv_unit_std = jnp.std(stats.raw_adv, axis=-1)
    return stats
