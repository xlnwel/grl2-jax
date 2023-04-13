from jax import lax, nn, random
import jax.numpy as jnp
import chex

from core.elements.loss import LossBase
from core.typing import dict2AttrDict
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
    def loss(
        self, 
        theta, 
        rng, 
        data, 
        name='theta',
    ):
        rngs = random.split(rng, 3)
        if data.dim_mask is None:
            dim_mask = jnp.ones_like(data.obs)
        else:
            dim_mask = jnp.zeros_like(data.obs) + data.dim_mask

        if self.model.config.model_norm_obs:
            obs = normalize(
                data.obs, 
                data.obs_loc, 
                data.obs_scale, 
                dim_mask=dim_mask, 
                np=jnp
            )
            next_obs = normalize(
                data.next_obs, 
                data.obs_loc, 
                data.obs_scale, 
                dim_mask=dim_mask, 
                np=jnp
            )
        else:
            obs = data.obs
            next_obs = data.next_obs

        # observation loss
        dist = self.modules.emodels(
            theta.emodels, rngs[0], obs, data.action, training=True
        )
        stats = dict2AttrDict(dist.get_stats('model'), to_copy=True)
        stats.norm_obs = obs
        
        if isinstance(dist, jax_dist.MultivariateNormalDiag):
            # for continuous obs, we predict 𝛥(o)
            model_target = expand_ensemble_dim(next_obs - obs, self.config.n_models)
        else:
            next_obs_ensemble = expand_ensemble_dim(next_obs, self.config.n_models)
            model_target = jnp.array(next_obs_ensemble, dtype=jnp.int32)
        stats.model_target = model_target

        edim_mask = jnp.zeros_like(model_target) + dim_mask
        model_loss, stats = compute_model_loss(
            self.config, dist, model_target, stats, 
            edim_mask, scale=data.obs_scale if self.model.config.model_norm_obs else None
        )

        # we use the predicted obs to predict the reward and discount
        pred_obs = lax.stop_gradient(dist.mode())
        if isinstance(dist, jax_dist.MultivariateNormalDiag):
            pred_obs = obs + pred_obs

        # reward loss
        reward_obs = self.model.get_reward_obs(dim_mask, data.obs, obs)
        reward_dist = self.modules.reward(
            theta.reward, rngs[1], reward_obs, data.action)
        reward_loss, stats = compute_reward_loss(
            self.config, reward_dist, data.reward, stats)

        # discount loss
        discount_dist = self.modules.discount(
            theta.discount, rngs[2], obs, data.action)
        discount_loss, stats = compute_discount_loss(
            self.config, discount_dist, data.discount, stats)

        if isinstance(dist, jax_dist.MultivariateNormalDiag):
            next_obs = jnp.zeros_like(model_target) + data.next_obs
            if self.model.config.model_norm_obs:
                pred_obs = denormalize(
                    pred_obs, 
                    data.obs_loc, 
                    data.obs_scale, 
                    dim_mask=dim_mask, 
                    np=jnp
                )
            stats.trans_mae = jnp.where(
                dim_mask, lax.abs(next_obs - pred_obs), 0.)
        else:
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
                    dim_mask, lax.abs(next_obs - pred_obs), 0.)
            else:
                stats.trans_mae = stats.model_mae

        loss = model_loss + reward_loss + discount_loss
        if data.is_ratio is not None:
            loss = data.is_ratio * loss
        stats.loss = loss
        loss = jnp.mean(loss)

        model_priority = jax_math.mask_mean(
            stats.trans_mae, mask=edim_mask, 
            axis=utils.except_axis(stats.trans_mae, SAMPLE_AXIS))
        reward_priority = jnp.mean(stats.reward_mae, axis=[1, 2])
        discount_priority = jnp.mean(stats.discount_mae, axis=[1, 2])
        chex.assert_rank([model_priority, reward_priority, discount_priority], 1)

        stats.model_priority = model_priority
        stats.reward_priority = reward_priority
        stats.discount_priority = discount_priority
        stats.priority = (
            self.config.model_prio_coef * model_priority 
            + self.config.reward_prio_coef * reward_priority 
            + self.config.discount_prio_coef * discount_priority
        )

        return loss, stats


def create_loss(config, model, name='model'):
    loss = Loss(config=config, model=model, name=name)

    return loss


def compute_model_loss(
    config, dist, model_target, stats, dim_mask, scale=None, 
):
    n = jax_math.count_masks(
        dim_mask, axis=utils.except_axis(dim_mask, [ENSEMBLE_AXIS, SAMPLE_AXIS])
    )
    if config.model_loss_type == 'mbpo':
        mean_loss, var_loss = jax_loss.mbpo_model_loss(
            dist.loc, 
            lax.log(dist.scale_diag) * 2., 
            model_target
        )
        stats.model_mae = jnp.where(
            dim_mask, lax.abs(dist.loc - model_target), 0.)
        mean_loss = jax_math.mask_mean(
            mean_loss, mask=dim_mask, n=n, 
            axis=utils.except_axis(mean_loss, [ENSEMBLE_AXIS, SAMPLE_AXIS])
        )
        var_loss = jax_math.mask_mean(
            var_loss, mask=dim_mask, n=n, 
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
            loss, mask=dim_mask, n=n, 
            axis=utils.except_axis(loss, [ENSEMBLE_AXIS, SAMPLE_AXIS])
        )
    elif config.model_loss_type == 'mse':
        stats.model_mae = jnp.where(
            dim_mask, lax.abs(dist.loc - model_target), 0.)
        if scale is None or not config.pred_raw:
            mean = dist.loc
        else:
            mean = dist.loc * scale
            model_target = model_target * scale
        loss = .5 * (mean - model_target)**2
        loss = jax_math.mask_mean(
            loss, mask=dim_mask, n=n, 
            axis=utils.except_axis(loss, [ENSEMBLE_AXIS, SAMPLE_AXIS])
        )
    elif config.model_loss_type == 'discrete':
        loss = - dist.log_prob(model_target)
        pred_next_obs = dist.mode()
        obs_cons = jax_math.mask_mean(
            pred_next_obs == model_target, axis=-1
        )
        stats.obs_dim_consistency = jax_math.mask_mean(
            obs_cons, mask=dim_mask, n=n, 
        )
        stats.obs_consistency = jax_math.mask_mean(
            obs_cons == 1, mask=dim_mask, n=n, 
        )
        loss = jax_math.mask_mean(
            loss, mask=dim_mask, n=n, 
            axis=utils.except_axis(loss, [ENSEMBLE_AXIS, SAMPLE_AXIS])
        )
    else:
        raise NotImplementedError(config.model_loss_type)
    stats.emodel_loss = jnp.mean(loss, SAMPLE_AXIS)
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
