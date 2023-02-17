from jax import lax, nn, random
import jax.numpy as jnp
import chex
import optax

from core.elements.loss import LossBase
from core.typing import dict2AttrDict, AttrDict
from jax_tools import jax_dist, jax_loss


def ensemble_obs(obs, n_models):
    ensemble_obs = jnp.expand_dims(obs, -2)
    ensemble_obs = jnp.tile(ensemble_obs, [n_models, 1])
    return ensemble_obs


class Loss(LossBase):
    def loss(
        self, 
        theta, 
        rng, 
        data, 
        name='theta'
    ):
        next_obs_ensemble = ensemble_obs(data.next_obs, self.config.n_models)
        rngs = random.split(rng, 3)
        dist = self.modules.emodels(
            theta.emodels, rngs[0], data.obs, data.action, 
        )
        if isinstance(dist, jax_dist.MultivariateNormalDiag):
            # for continuous obs, we predict 𝛥(o)
            obs_ensemble = ensemble_obs(data.obs, self.config.n_models)
            pred_ensemble = next_obs_ensemble - obs_ensemble
        else:
            pred_ensemble = jnp.array(next_obs_ensemble, dtype=jnp.int32)

        stats = dict2AttrDict(dist.get_stats('model'), to_copy=True)

        model_loss, stats = compute_model_loss(
            self.config, pred_ensemble, stats)
        reward_dist = self.modules.reward(
            theta.reward, rngs[1], data.obs, data.action)
        reward_loss, stats = compute_reward_loss(
            self.config, reward_dist, data.reward, stats)
        discount_dist = self.modules.discount(
            theta.discount, rngs[2], data.next_obs)
        discount_loss, stats = compute_discount_loss(
            self.config, discount_dist, data.discount, stats)

        loss = model_loss + reward_loss + discount_loss
        stats.loss = loss
        stats.elite_indices = jnp.argsort(stats.mean_loss)

        return loss, stats


def create_loss(config, model, name='model'):
    loss = Loss(config=config, model=model, name=name)

    return loss


def compute_model_loss(
    config, pred_obs, stats
):
    if config.model_loss_type == 'mbpo':
        mean_loss, var_loss = jax_loss.mbpo_model_loss(
            stats.model_loc, 
            lax.log(stats.model_scale) * 2., 
            pred_obs
        )
        stats.model_mae = lax.abs(stats.model_loc - pred_obs)
        mean_loss = jnp.mean(mean_loss, [0, 1, 2])
        var_loss = jnp.mean(var_loss, [0, 1, 2])
        stats.mean_loss = mean_loss
        stats.var_loss = var_loss
        loss = jnp.sum(mean_loss) + jnp.sum(var_loss)
        chex.assert_rank([mean_loss, var_loss], 1)
    elif config.model_loss_type == 'mse':
        loss = .5 * (stats.model_loc - pred_obs) ** 2
        stats.mean_loss = jnp.mean(loss, [0, 1, 2, 4])
        loss = jnp.sum(stats.mean_loss)
    elif config.model_loss_type == 'discrete':
        loss = optax.softmax_cross_entropy_with_integer_labels(
            stats.model_logits, pred_obs)
        pred_next_obs = jnp.argmax(stats.model_logits, -1)
        stats.obs_consistency = jnp.mean(pred_next_obs == pred_obs)
        stats.mean_loss = jnp.mean(loss, [0, 1, 2, 4])
        assert len(stats.mean_loss.shape) == 1, stats.mean_loss.shape
        loss = jnp.sum(stats.mean_loss)
    else:
        raise NotImplementedError
    stats.model_loss = config.model_coef * loss

    return loss, stats


def compute_reward_loss(
    config, reward_dist, reward, stats
):
    pred_reward = reward_dist.mode()
    reward_loss = jnp.mean(.5 * (pred_reward - reward)**2)
    stats.pred_reward = pred_reward
    stats.reward_mae = lax.abs(pred_reward - reward)
    stats.reward_consistency = jnp.mean(stats.reward_mae < .1)
    stats.reward_loss = config.reward_coef * reward_loss

    return reward_loss, stats


def compute_discount_loss(
    config, discount_dist, discount, stats
):
    discount_loss = - discount_dist.log_prob(discount)
    discount_loss = jnp.mean(discount_loss)
    discount = discount_dist.mode()
    stats.pred_discount = discount
    stats.discount_consistency = jnp.mean(stats.discount == discount)
    stats.discount_loss = config.discount_coef * discount_loss

    return discount_loss, stats
