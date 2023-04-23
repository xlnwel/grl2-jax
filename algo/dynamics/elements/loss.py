from jax import lax, nn, random
import jax.numpy as jnp
import chex

from core.elements.loss import LossBase
from core.typing import AttrDict, dict2AttrDict
from jax_tools import jax_dist, jax_math, jax_loss
from tools.rms import normalize, denormalize
from tools import utils
from algo.dynamics.elements.nn import ENSEMBLE_AXIS


SAMPLE_AXIS = 1


def expand_ensemble_dim(x, n_models):
    assert ENSEMBLE_AXIS == 0, ENSEMBLE_AXIS
    ex = jnp.expand_dims(x, ENSEMBLE_AXIS)
    ex = jnp.tile(ex, [n_models, *[1 for _ in range(x.ndim)]])
    return ex


class Loss(LossBase):
    def model_loss(
        self, 
        theta, 
        rng, 
        data, 
    ):
        dim_mask = data.dim_mask

        dist = self.modules.emodels(
            theta, rng, data.norm_obs, data.action, training=True
        )
        stats = dict2AttrDict(dist.get_stats('model'), to_copy=True)


        if isinstance(dist, jax_dist.MultivariateNormalDiag):
            # for continuous obs, we predict 𝛥(o)
            if self.model.config.pred_raw:
                model_target = data.next_obs - data.obs
            else:
                model_target = data.next_norm_obs - data.norm_obs
            model_target = expand_ensemble_dim(model_target, self.config.n_models)
        else:
            assert self.config.model_loss_type == 'discrete', self.config.model_loss_type
            next_obs_ensemble = expand_ensemble_dim(data.next_obs, self.config.n_models)
            model_target = jnp.array(next_obs_ensemble, dtype=jnp.int32)
        
        edim_mask = jnp.zeros_like(model_target) + dim_mask
        loss, stats = compute_model_loss(
            self.config, dist, model_target, stats, edim_mask,
        )

        if data.is_ratio is not None:
            loss = data.is_ratio * loss
        loss = jnp.mean(loss)

        if isinstance(dist, jax_dist.MultivariateNormalDiag):
            if self.model.config.pred_raw:
                pred_obs = data.obs + dist.loc
            else:
                pred_obs = data.norm_obs + dist.loc
                if self.model.config.model_norm_obs:
                    pred_obs = denormalize(
                        pred_obs, 
                        data.obs_loc, 
                        data.obs_scale, 
                        dim_mask=dim_mask, 
                        np=jnp
                    )
            next_obs = jnp.zeros_like(model_target) + data.next_obs
            stats.trans_mae = jnp.where(
                edim_mask, lax.abs(next_obs - pred_obs), 0.)
        else:
            assert not self.model.config.model_norm_obs
            stats.trans_mae = stats.model_mae

        stats.model_priority = jax_math.mask_mean(
            stats.trans_mae, mask=edim_mask, 
            axis=utils.except_axis(stats.trans_mae, SAMPLE_AXIS))

        return loss, stats

    def reward_loss(
        self, 
        theta, 
        rng, 
        data, 
    ):
        dim_mask = data.dim_mask

        reward_obs = self.model.get_reward_obs(dim_mask, data.obs, data.norm_obs)
        reward_dist = self.modules.reward(
            theta, rng, reward_obs, data.action)
        loss, stats = compute_reward_loss(
            self.config, reward_dist, data.reward, AttrDict())

        if data.is_ratio is not None:
            loss = data.is_ratio * loss
        loss = jnp.mean(loss)

        stats.reward_priority = jnp.mean(stats.reward_mae, axis=[1, 2])

        return loss, stats

    def discount_loss(
        self, 
        theta, 
        rng, 
        data, 
    ):
        dim_mask = data.dim_mask

        discount_obs = self.model.get_discount_obs(dim_mask, data.norm_obs)
        discount_dist = self.modules.discount(
            theta, rng, discount_obs, data.action)
        loss, stats = compute_discount_loss(
            self.config, discount_dist, data.discount, AttrDict())

        if data.is_ratio is not None:
            loss = data.is_ratio * loss
        loss = jnp.mean(loss)

        stats.discount_priority = jnp.mean(stats.discount_mae, axis=[1, 2])

        return loss, stats

    def loss(
        self, 
        theta, 
        rng, 
        data, 
        name='theta',
    ):
        rngs = random.split(rng, 3)

        model_loss, model_stats = self.model_loss(theta.emodels, rngs[0], data)
        reward_loss, reward_stats = self.reward_loss(theta.reward, rngs[1], data)
        discount_loss, discount_stats = self.discount_loss(theta.discount, rngs[2], data)
        stats = self.combine_stats(model_stats, reward_stats, discount_stats)

        loss = model_loss + reward_loss + discount_loss
        if data.is_ratio is not None:
            loss = data.is_ratio * loss
        stats.loss = loss
        loss = jnp.mean(loss)

        return loss, stats
    
    def combine_stats(self, stats, reward_stats, discount_stats):
        stats.update(reward_stats)
        stats.update(discount_stats)
        stats.priority = (
            self.config.model_prio_coef * stats.model_priority 
            + self.config.reward_prio_coef * stats.reward_priority 
            + self.config.discount_prio_coef * stats.discount_priority
        )
        return stats



def create_loss(config, model, name='model'):
    loss = Loss(config=config, model=model, name=name)

    return loss


def compute_model_loss(
    config, dist, model_target, stats, dim_mask, 
):
    if config.model_loss_type == 'mbpo':
        mean_loss, var_loss = jax_loss.mbpo_model_loss(
            dist.loc, 
            lax.log(dist.scale_diag) * 2., 
            model_target
        )
        stats.model_mae = jnp.where(
            dim_mask, lax.abs(dist.loc - model_target), 0.)
        mean_loss = jnp.mean(
            mean_loss, 
            axis=utils.except_axis(mean_loss, [ENSEMBLE_AXIS, SAMPLE_AXIS])
        )
        var_loss = jnp.mean(
            var_loss, 
            axis=utils.except_axis(var_loss, [ENSEMBLE_AXIS, SAMPLE_AXIS])
        )
        stats.mean_loss = mean_loss
        stats.var_loss = var_loss
        loss = mean_loss + var_loss
    elif config.model_loss_type == 'ce':
        loss = - dist.log_prob(model_target)
        stats.model_mae = jnp.where(
            dim_mask, lax.abs(dist.loc - model_target), 0.)
        loss = jax_math.mask_mean(
            loss, 
            axis=utils.except_axis(loss, [ENSEMBLE_AXIS, SAMPLE_AXIS])
        )
    elif config.model_loss_type == 'mse':
        stats.model_mae = jnp.where(
            dim_mask, lax.abs(dist.loc - model_target), 0.)
        loss = .5 * (dist.loc - model_target)**2
        loss = jax_math.mask_mean(
            loss, mask=dim_mask, 
            axis=utils.except_axis(loss, [ENSEMBLE_AXIS, SAMPLE_AXIS])
        )
    elif config.model_loss_type == 'discrete':
        loss = - dist.log_prob(model_target)
        pred_next_obs = dist.mode()
        # obs_cons = jax_math.mask_mean(
        #     pred_next_obs == model_target, axis=-1
        # )
        # stats.obs_dim_consistency = jax_math.mask_mean(
        #     obs_cons, mask=dim_mask, 
        # )
        # stats.obs_consistency = jax_math.mask_mean(
        #     obs_cons == 1, mask=dim_mask, 
        # )
        loss = jax_math.mask_mean(
            loss, mask=dim_mask, 
            axis=utils.except_axis(loss, [ENSEMBLE_AXIS, SAMPLE_AXIS])
        )
    else:
        raise NotImplementedError(config.model_loss_type)
    stats.emodel_metrics = jnp.mean(loss, SAMPLE_AXIS)
    loss = config.model_coef * jnp.sum(loss, ENSEMBLE_AXIS)
    stats.model_loss = loss

    return loss, stats


def compute_reward_loss(
    config, reward_dist, reward, stats
):
    raw_reward_loss = .5 * (reward_dist.loc - reward)**2
    pred_reward = reward_dist.mode()
    stats.pred_reward = pred_reward
    stats.reward_mae = lax.abs(pred_reward - reward)
    stats.reward_consistency = jnp.mean(
        stats.reward_mae < .1 * (jnp.max(reward) - jnp.min(reward)))
    stats.scaled_reward_loss, reward_loss = jax_loss.to_loss(
        raw_reward_loss, config.reward_coef, 
        axis=utils.except_axis(raw_reward_loss, SAMPLE_AXIS)
    )
    stats.reward_loss = reward_loss

    return reward_loss, stats


def compute_discount_loss(
    config, discount_dist, discount, stats
):
    raw_discount_loss = - discount_dist.log_prob(discount)
    pred_discount = discount_dist.mode()
    stats.pred_discount = pred_discount
    stats.discount_mae = lax.abs(pred_discount - discount)
    discount_self_cons = jnp.all(discount == pred_discount, -1)
    stats.discount_self_consistency = jnp.mean(discount_self_cons)
    stats.discount_consistency = jnp.mean(discount == pred_discount)
    stats.scaled_discount_loss, discount_loss = jax_loss.to_loss(
        raw_discount_loss, config.discount_coef, 
        axis=utils.except_axis(raw_discount_loss, SAMPLE_AXIS)
    )
    stats.discount_loss = discount_loss

    return discount_loss, stats
