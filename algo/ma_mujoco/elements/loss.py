from jax import lax, nn, random
import jax.numpy as jnp
import chex
import optax

from core.elements.loss import LossBase
from core.typing import dict2AttrDict, AttrDict
from jax_tools import jax_dist, jax_loss, jax_math, jax_utils
from tools.utils import prefix_name
from tools.display import print_dict_info


class Loss(LossBase):
    def loss(
        self, 
        theta, 
        rng, 
        data, 
        name='theta'
    ):
        obs, next_obs = jax_utils.split_data(data.obs, data.next_obs, 1)
        ensemble_next_obs = jnp.expand_dims(next_obs, -2)
        ensemble_next_obs = jnp.tile(ensemble_next_obs, [self.config.n, 1])
        ensemble_next_obs = jnp.array(ensemble_next_obs, dtype=jnp.int32)
        dist = self.modules.emodels(
            theta.emodels, rng, obs, data.action, 
        )

        stats = dict2AttrDict(dist.get_stats('model'), to_copy=True)

        loss, stats = compute_model_loss(self.config, ensemble_next_obs, stats)
        rng = random.split(rng, 1)[0]
        reward_dist = self.modules.reward(theta.reward, rng, obs, data.action)
        reward_loss, stats = compute_reward_loss(
            self.config, reward_dist, data.reward, stats)
        loss = loss + reward_loss
        stats.loss = loss
        stats.elite_indices = jnp.argsort(stats.mean_loss)
        stats = prefix_name(stats, name, filter=['elite_indices'])

        return loss, stats


def create_loss(config, model, name='model'):
    loss = Loss(config=config, model=model, name=name)

    return loss


def compute_model_loss(
    config, 
    next_obs, 
    stats
):
    if config.model_loss_type == 'mbpo':
        mean_loss, var_loss = jax_loss.mbpo_model_loss(
            stats.model_loc, 
            lax.log(stats.model_scale) * 2., 
            next_obs
        )

        mean_loss = jnp.mean(mean_loss, [0, 1, 2])
        var_loss = jnp.mean(var_loss, [0, 1, 2])
        stats.mean_loss = mean_loss
        stats.var_loss = var_loss
        loss = jnp.sum(mean_loss) + jnp.sum(var_loss)
        chex.assert_rank([mean_loss, var_loss], 1)
    elif config.model_loss_type == 'mse':
        loss = (stats.model_loc - next_obs) ** 2
        stats.mean_loss = jnp.mean(loss, [0, 1, 2, 4])
        loss = jnp.mean(stats.mean_loss)
    elif config.model_loss_type == 'discrete':
        loss = optax.softmax_cross_entropy_with_integer_labels(
            stats.model_logits, next_obs)
        pred_next_obs = jnp.argmax(stats.model_logits, -1)
        stats.obs_consistency = jnp.mean(pred_next_obs == next_obs)
        stats.mean_loss = jnp.mean(loss, [0, 1, 2, 4])
        assert len(stats.mean_loss.shape) == 1, stats.mean_loss.shape
        loss = jnp.mean(stats.mean_loss)
    else:
        raise NotImplementedError
    stats.model_loss = loss

    return loss, stats

def compute_reward_loss(
    config, reward_dist, reward, stats
):
    pred_reward = jnp.squeeze(reward_dist.mode(), -1)
    reward_loss = jnp.mean(.5 * (pred_reward - reward)**2)
    stats.pred_reward = pred_reward
    stats.reward_mae = jnp.abs(pred_reward - reward)
    stats.reward_consistency = jnp.mean(stats.reward_mae < .1)
    stats.reward_loss = reward_loss

    return reward_loss, stats
