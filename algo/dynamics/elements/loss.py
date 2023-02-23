from jax import lax, nn, random
import jax.numpy as jnp
import chex

from core.elements.loss import LossBase
from core.typing import dict2AttrDict
from jax_tools import jax_dist, jax_loss


def ensemble_obs(obs, n_models):
    eobs = jnp.expand_dims(obs, -2)
    eobs = jnp.tile(eobs, [n_models, 1])
    return eobs


class Loss(LossBase):
    def loss(
        self, 
        theta, 
        rng, 
        data, 
        name='theta'
    ):
        rngs = random.split(rng, 3)
        dist = self.modules.emodels(
            theta.emodels, rngs[0], data.obs, data.action, 
        )
        if isinstance(dist, jax_dist.MultivariateNormalDiag):
            # for continuous obs, we predict 𝛥(o)
            pred_ensemble = ensemble_obs(
                data.next_obs - data.obs, self.config.n_models)
        else:
            next_obs_ensemble = ensemble_obs(
                data.next_obs, self.config.n_models)
            pred_ensemble = jnp.array(next_obs_ensemble, dtype=jnp.int32)

        stats = dict2AttrDict(dist.get_stats('model'), to_copy=True)

        model_loss, stats = compute_model_loss(
            self.config, dist, pred_ensemble, stats)
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
    config, dist, pred_obs, stats
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
    elif config.model_loss_type == 'ce':
        loss = - dist.log_prob(pred_obs)
        stats.mean_loss = jnp.mean(loss, [0, 1, 2])
        stats.model_mae = lax.abs(stats.model_loc - pred_obs)
        chex.assert_rank([stats.mean_loss], 1)
        loss = jnp.sum(stats.mean_loss)
    elif config.model_loss_type == 'discrete':
        loss = - dist.log_prob(pred_obs)
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
    raw_reward_loss = .5 * (pred_reward - reward)**2
    stats.pred_reward = pred_reward
    stats.reward_mae = lax.abs(pred_reward - reward)
    stats.reward_consistency = jnp.mean(stats.reward_mae < .1)
    stats.scaled_reward_loss, reward_loss = jax_loss.to_loss(
        raw_reward_loss, 
        config.reward_coef, 
    )
    stats.reward_loss = reward_loss

    return reward_loss, stats


def compute_discount_loss(
    config, discount_dist, discount, stats
):
    raw_discount_loss = - discount_dist.log_prob(discount)
    discount = discount_dist.mode()
    stats.pred_discount = discount
    stats.discount_mae = lax.abs(discount_dist.mode() - discount)
    stats.discount_consistency = jnp.mean(stats.discount == discount)
    stats.scaled_discount_loss, discount_loss = jax_loss.to_loss(
        raw_discount_loss, 
        config.discount_coef, 
    )
    stats.discount_loss = discount_loss

    return discount_loss, stats
