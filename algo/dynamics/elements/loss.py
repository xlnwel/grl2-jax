from jax import lax, nn, random
import jax.numpy as jnp
import chex

from core.elements.loss import LossBase
from core.typing import dict2AttrDict
from jax_tools import jax_dist, jax_math, jax_loss
from tools.rms import normalize, denormalize
from tools import utils
from algo.dynamics.elements.nn import ENSEMBLE_AXIS


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
            norm_obs = normalize(
                data.obs, 
                data.obs_loc, 
                data.obs_scale, 
                dim_mask=dim_mask, 
                np=jnp
            )
            next_norm_obs = normalize(
                data.next_obs, 
                data.obs_loc, 
                data.obs_scale, 
                dim_mask=dim_mask, 
                np=jnp
            )
        else:
            norm_obs = data.obs
            next_norm_obs = data.next_obs

        # observation loss
        model_dist, reward_dist, disc_dist = self.modules.edynamics(
            theta.edynamics, rngs[0], norm_obs, data.action, training=True
        )
        stats = dict2AttrDict(model_dist.get_stats('model'), to_copy=True)
        stats.norm_obs = norm_obs
        
        if isinstance(model_dist, jax_dist.MultivariateNormalDiag):
            # for continuous obs, we predict 𝛥(o)
            model_target = expand_ensemble_dim(
                next_norm_obs - norm_obs, self.config.n_models)
        else:
            next_obs_ensemble = expand_ensemble_dim(
                next_norm_obs, self.config.n_models)
            model_target = jnp.array(next_obs_ensemble, dtype=jnp.int32)
        stats.model_target = model_target

        dim_mask = jnp.zeros_like(model_target) + dim_mask
        model_loss, stats = compute_model_loss(
            self.config, model_dist, model_target, stats, dim_mask)

        # we use the predicted obs to predict the reward and discount
        pred_obs = lax.stop_gradient(model_dist.mode())
        if isinstance(model_dist, jax_dist.MultivariateNormalDiag):
            pred_obs = norm_obs + pred_obs
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
            stats.trans_mae = stats.model_mae

        # reward loss
        reward_loss, stats = compute_reward_loss(
            self.config, reward_dist, data.reward, stats)

        # discount loss
        discount_loss, stats = compute_discount_loss(
            self.config, disc_dist, data.discount, stats)

        loss = model_loss + reward_loss + discount_loss
        stats.loss = loss

        return loss, stats


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
            mean_loss, axis=utils.except_axis(mean_loss, ENSEMBLE_AXIS)
        )
        var_loss = jnp.mean(
            var_loss, axis=utils.except_axis(var_loss, ENSEMBLE_AXIS)
        )
        stats.mean_loss = mean_loss
        stats.var_loss = var_loss
        loss = jnp.sum(mean_loss) + jnp.sum(var_loss)
        chex.assert_rank([mean_loss, var_loss], 1)
    elif config.model_loss_type == 'ce':
        loss = - dist.log_prob(model_target)
        stats.model_mae = jnp.where(
            dim_mask, lax.abs(dist.loc - model_target), 0.)
        stats.mean_loss = jnp.mean(
            loss, axis=utils.except_axis(loss, ENSEMBLE_AXIS)
        )
        chex.assert_rank([stats.mean_loss], 1)
        loss = jnp.sum(stats.mean_loss)
    elif config.model_loss_type == 'mse':
        loss = .5 * (dist.loc - model_target)**2
        stats.model_mae = jnp.where(
            dim_mask, lax.abs(dist.loc - model_target), 0.)
        stats.mean_loss = jax_math.mask_mean(
            loss, mask=dim_mask, 
            axis=utils.except_axis(loss, ENSEMBLE_AXIS)
        )
        chex.assert_rank([stats.mean_loss], 1)
        loss = jnp.sum(stats.mean_loss)
    elif config.model_loss_type == 'discrete':
        loss = - dist.log_prob(model_target)
        pred_next_obs = dist.mode()
        obs_cons = jnp.mean(pred_next_obs == model_target, axis=-1)
        stats.obs_dim_consistency = jnp.mean(obs_cons)
        stats.obs_consistency = jnp.mean(obs_cons == 1)
        stats.mean_loss = jnp.mean(
            loss, axis=utils.except_axis(loss, ENSEMBLE_AXIS)
        )
        assert len(stats.mean_loss.shape) == 1, stats.mean_loss.shape
        loss = jnp.sum(stats.mean_loss)
    else:
        raise NotImplementedError(config.model_loss_type)
    stats.model_loss = config.model_coef * loss
    stats.emodel_metrics = stats.mean_loss

    return loss, stats


def compute_reward_loss(
    config, reward_dist, reward, stats
):
    raw_reward_loss = .5 * (reward_dist.loc - reward)**2
    pred_reward = reward_dist.mode()
    stats.pred_reward = pred_reward
    stats.reward_mae = lax.abs(pred_reward - reward)
    stats.reward_consistency = jnp.mean(
        stats.reward_mae < .1 * (jnp.max(reward) - jnp.min(reward))
    )
    stats.scaled_reward_loss, reward_loss = jax_loss.to_loss(
        raw_reward_loss, config.reward_coef, 
    )
    stats.reward_loss = reward_loss

    return reward_loss, stats


def compute_discount_loss(
    config, discount_dist, discount, stats
):
    raw_discount_loss = - discount_dist.log_prob(discount)
    stats.raw_discount_loss = raw_discount_loss
    pred_discount = discount_dist.mode()
    stats.pred_discount = pred_discount
    stats.discount_mae = lax.abs(pred_discount - discount)
    stats.discount_consistency = jnp.mean(discount == pred_discount)
    stats.scaled_discount_loss, discount_loss = jax_loss.to_loss(
        raw_discount_loss, config.discount_coef, 
    )
    stats.discount_loss = discount_loss

    return discount_loss, stats
