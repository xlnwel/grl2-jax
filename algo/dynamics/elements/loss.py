from jax import lax, nn, random
import jax.numpy as jnp
import chex

from core.elements.loss import LossBase
from core.typing import dict2AttrDict
from jax_tools import jax_dist, jax_loss
from tools import utils
from tools.rms import normalize, denormalize
from algo.dynamics.elements.nn import ENSEMBLE_AXIS


def ensemble_obs(obs, n_models):
    assert ENSEMBLE_AXIS == 0, ENSEMBLE_AXIS
    eobs = jnp.expand_dims(obs, ENSEMBLE_AXIS)
    eobs = jnp.tile(eobs, [n_models, *[1 for _ in range(obs.ndim)]])
    return eobs


class Loss(LossBase):
    def loss(
        self, 
        theta, 
        rng, 
        data, 
        name='theta',
    ):
        rngs = random.split(rng, 3)

        if self.model.config.model_norm_obs:
            data.obs = normalize(data.obs, data.obs_loc, data.obs_scale)
            data.next_obs = normalize(data.next_obs, data.obs_loc, data.obs_scale)

        dist = self.modules.emodels(
            theta.emodels, rngs[0], data.obs, data.action, training=True
        )
        stats = dict2AttrDict(dist.get_stats('model'), to_copy=True)
        
        if isinstance(dist, jax_dist.MultivariateNormalDiag):
            # for continuous obs, we predict 𝛥(o)
            model_target = ensemble_obs(
                data.next_obs - data.obs, self.config.n_models)
        else:
            next_obs_ensemble = ensemble_obs(
                data.next_obs, self.config.n_models)
            model_target = jnp.array(next_obs_ensemble, dtype=jnp.int32)

        model_loss, stats = compute_model_loss(
            self.config, dist, model_target, stats)
        reward_dist = self.modules.reward(
            theta.reward, rngs[1], data.obs, data.action, data.next_obs)
        reward_loss, stats = compute_reward_loss(
            self.config, reward_dist, data.reward, stats)
        discount_dist = self.modules.discount(
            theta.discount, rngs[2], data.next_obs)
        discount_loss, stats = compute_discount_loss(
            self.config, discount_dist, data.discount, stats)

        if isinstance(dist, jax_dist.MultivariateNormalDiag):
            pred_obs = data.obs + dist.loc
            next_obs = data.next_obs
            if self.model.config.model_norm_obs:
                pred_obs = denormalize(pred_obs, data.obs_loc, data.obs_scale)
                next_obs = denormalize(next_obs, data.obs_loc, data.obs_scale)
            stats.trans_mae = lax.abs(next_obs - pred_obs)
        else:
            stats.trans_mae = stats.model_mae

        loss = model_loss + reward_loss + discount_loss
        stats.loss = loss

        return loss, stats


def create_loss(config, model, name='model'):
    loss = Loss(config=config, model=model, name=name)

    return loss


def compute_model_loss(
    config, dist, model_target, stats
):
    if config.model_loss_type == 'mbpo':
        mean_loss, var_loss = jax_loss.mbpo_model_loss(
            stats.model_loc, 
            lax.log(stats.model_scale) * 2., 
            model_target
        )
        stats.model_mae = lax.abs(stats.model_loc - model_target)
        mean_loss = jnp.mean(mean_loss, utils.except_axis(mean_loss, ENSEMBLE_AXIS))
        var_loss = jnp.mean(var_loss, utils.except_axis(var_loss, ENSEMBLE_AXIS))
        stats.mean_loss = mean_loss
        stats.var_loss = var_loss
        loss = jnp.sum(mean_loss) + jnp.sum(var_loss)
        chex.assert_rank([mean_loss, var_loss], 1)
    elif config.model_loss_type == 'ce':
        loss = - dist.log_prob(model_target)
        stats.mean_loss = jnp.mean(loss, utils.except_axis(loss, ENSEMBLE_AXIS))
        stats.model_mae = lax.abs(stats.model_loc - model_target)
        chex.assert_rank([stats.mean_loss], 1)
        loss = jnp.sum(stats.mean_loss)
    elif config.model_loss_type == 'mse':
        loss = .5 * (stats.model_loc - model_target)**2
        stats.model_mae = lax.abs(stats.model_loc - model_target)
        stats.mean_loss = jnp.mean(loss, utils.except_axis(loss, ENSEMBLE_AXIS))
        chex.assert_rank([stats.mean_loss], 1)
        loss = jnp.sum(stats.mean_loss)
    elif config.model_loss_type == 'discrete':
        loss = - dist.log_prob(model_target)
        pred_next_obs = dist.mode()
        obs_cons = jnp.mean(pred_next_obs == model_target, -1)
        stats.obs_consistency = jnp.mean(obs_cons == 1)
        stats.mean_loss = jnp.mean(loss, utils.except_axis(loss, ENSEMBLE_AXIS))
        assert len(stats.mean_loss.shape) == 1, stats.mean_loss.shape
        loss = jnp.sum(stats.mean_loss)
    else:
        raise NotImplementedError
    stats.model_loss = config.model_coef * loss

    return loss, stats


def compute_reward_loss(
    config, reward_dist, reward, stats
):
    raw_reward_loss = - reward_dist.log_prob(reward)
    pred_reward = reward_dist.mode()
    stats.pred_reward = pred_reward
    stats.reward_mae = lax.abs(pred_reward - reward)
    stats.reward_consistency = jnp.mean(
        stats.reward_mae < .1 * (jnp.max(reward) - jnp.min(reward)))
    stats.scaled_reward_loss, reward_loss = jax_loss.to_loss(
        raw_reward_loss, config.reward_coef, 
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
    stats.discount_consistency = jnp.mean(discount == pred_discount)
    stats.scaled_discount_loss, discount_loss = jax_loss.to_loss(
        raw_discount_loss, config.discount_coef, 
    )
    stats.discount_loss = discount_loss

    return discount_loss, stats
