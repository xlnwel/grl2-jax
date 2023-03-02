from jax import lax, nn, random
import jax.numpy as jnp

from core.elements.loss import LossBase
from core.typing import dict2AttrDict
from jax_tools import jax_dist
from algo.dynamics.elements.loss import ensemble_obs, compute_model_loss, \
    compute_reward_loss, compute_discount_loss


class Loss(LossBase):
    def loss(
        self, 
        theta, 
        rng, 
        data, 
        name='theta', 
        **kwargs
    ):
        rngs = random.split(rng, 3)
        if self.config.obs_normalization:
            dist, stats = self.model.normalized_emodels(
                theta, rngs[0], data.obs, data.action,
                **kwargs
            )
        else:
            dist = self.modules.emodels(
                theta.emodels, rngs[0], data.obs, data.action, training=True
            )
            stats = dict2AttrDict(dist.get_stats('model'), to_copy=True)

        if isinstance(dist, jax_dist.MultivariateNormalDiag):
            # for continuous obs, we predict 𝛥(o)
            pred_ensemble = ensemble_obs(
                data.next_obs - data.obs, self.config.n_models)
        else:
            next_obs_ensemble = ensemble_obs(
                data.next_obs, self.config.n_models)
            pred_ensemble = jnp.array(next_obs_ensemble, dtype=jnp.int32)

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

        return loss, stats


def create_loss(config, model, name='model'):
    loss = Loss(config=config, model=model, name=name)

    return loss
