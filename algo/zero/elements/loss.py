from jax import lax, nn, random
import jax.numpy as jnp
import chex

from core.elements.loss import LossBase
from core.typing import dict2AttrDict
from jax_tools import jax_loss, jax_math, jax_utils
from tools.utils import prefix_name
from .utils import compute_values, compute_policy_dist, compute_policy, compute_next_obs_dist


ACTIONS = [
    [0, -1],  # Move left
    [0, 1],  # Move right
    [-1, 0],  # Move up
    [1, 0],  # Move down
    [0, 0]  # don't move
]


class Loss(LossBase):
    def loss(
        self, 
        theta, 
        rng, 
        data, 
        name='train', 
    ):
        rngs = random.split(rng, 2)
        stats = dict2AttrDict(self.config.stats)

        if data.action_mask is not None:
            stats.n_avail_actions = jnp.sum(data.action_mask, -1)
        if data.sample_mask is not None:
            stats.n_alive_units = jnp.sum(data.sample_mask, -1)

        stats.value, next_value = compute_values(
            self.modules.value, 
            theta.value, 
            rngs[0], 
            data.global_state, 
            data.next_global_state, 
            sid=data.sid, 
            next_sid=data.next_sid, 
            idx=data.idx, 
            next_idx=data.next_idx, 
            event=data.event, 
            next_event=data.next_event, 
            seq_axis=1, 
        )

        act_dist, stats.pi_logprob, stats.log_ratio, stats.ratio = \
            compute_policy(
                self.modules.policy, 
                theta.policy, 
                rngs[1], 
                data.obs, 
                data.next_obs, 
                data.action, 
                data.mu_logprob, 
                sid=data.sid, 
                next_sid=data.next_sid, 
                idx=data.idx, 
                next_idx=data.next_idx, 
                event=data.event, 
                next_event=data.next_event, 
                action_mask=data.action_mask, 
                next_action_mask=data.next_action_mask, 
                seq_axis=1, 
            )
        stats = record_policy_stats(data, stats, act_dist)

        v_target, stats.raw_adv = jax_loss.compute_target_advantage(
            config=self.config, 
            reward=data.reward, 
            discount=data.discount, 
            reset=data.reset, 
            value=lax.stop_gradient(stats.value), 
            next_value=next_value, 
            ratio=lax.stop_gradient(stats.ratio), 
            gamma=stats.gamma, 
            lam=stats.lam, 
            axis=1
        )
        stats.v_target = lax.stop_gradient(v_target)
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
        stats.advantage = lax.stop_gradient(stats.advantage)

        actor_loss, stats = compute_actor_loss(
            self.config, 
            data, 
            stats, 
            act_dist=act_dist, 
        )
        value_loss, stats = compute_vf_loss(
            self.config, 
            data, 
            stats, 
        )
        loss = actor_loss + value_loss

        # obs, data.next_obs = jax_utils.split_data(data.obs, data.next_obs, 1)
        # dist = compute_next_obs_dist(
        #     self.modules.models, 
        #     theta.models, 
        #     rngs[2], 
        #     obs, 
        #     data.action, 
        # )
        # stats.model_mean = dist.mean
        # stats.model_logvar = dist.logstd * 2

        # mean_loss, var_loss = compute_model_loss(self.config, data, stats)
        # stats.mean_loss = mean_loss
        # stats.var_loss = var_loss
        # loss = loss + jnp.sum(mean_loss) + jnp.sum(var_loss)

        stats = prefix_name(stats, name)
        # stats.elite_idx = jnp.argsort(mean_loss)

        return loss, stats

    def imagine_ahead(
        self, 
        theta, 
        rng, 
        data, 
    ):
        def monster_hunter_step(obs, action1, action2):
            a1_pos = obs[:2]
            a2_pos = obs[2:4]
            a1_pos = jnp.clip(a1_pos + ACTIONS[action1], 0, 5)
            a2_pos = jnp.clip(a2_pos + ACTIONS[action2], 0, 5)
            
            monster_pos = obs[4:6]
            apple1_pos = obs[6:8]
            apple2_pos = obs[8:]
            reward1 = jnp.where(
                jnp.all(a1_pos == apple1_pos) or jnp.all(a1_pos == apple2_pos), 
                2., 0.
            )
            a1_monster = jnp.all(a1_pos == monster_pos)
            a2_monster = jnp.all(a2_pos == monster_pos)
            reward1 = jnp.where(
                a1_monster and a2_monster, 
                5., jnp.where(a1_monster, -2, reward1
                )
            )

            reward2 = jnp.where(
                jnp.all(a2_pos == apple1_pos) or jnp.all(a2_pos == apple2_pos), 
                2., 0.
            )
            reward2 = jnp.where(
                a1_monster and a2_monster,  
                5., jnp.where(a2_monster, -2, reward2)
            )
            
            obs = jnp.stack([
                jnp.concatenate(a1_pos, a2_pos, monster_pos, apple1_pos, apple2_pos),
                jnp.concatenate(a2_pos, a1_pos, monster_pos, apple1_pos, apple2_pos)
            ])
            reward = jnp.stack(reward1, reward2)
            return obs, reward, jnp.ones_like(reward)


        obs = data.obs
        obs_list = [obs]
        action_list = []
        for _ in range(self.config.rollout_length):
            # rng, rng2 = random.split(rng, 2)
            # act_dist = compute_policy_dist(
            #     self.modules.policy, 
            #     theta.policy, 
            #     rng, 
            # )
            # action_list.append(act_dist.sample())
            # model_params = random.choice(theta['emodels'])
            # obs_dist = compute_next_obs_dist(
            #     self.modules.model, 
            #     model_params, 
            #     rng2, 
            #     obs_list[-1],
            #     action_list[-1]
            # )
            # obs_list.append(obs_dist.sample())
            rng = random.split(rng, 1)[1]
            assert obs.shape[-2] == 2, obs.shape
            act_dist1 = compute_policy_dist(
                self.modules.policy, 
                theta.policy, 
                rng, 
                obs[..., 0, :], 
                None, 
                0, 
                None
            )
            action1 = act_dist1.sample()
            act_dist2 = compute_policy_dist(
                self.modules.policy, 
                theta.policy, 
                rng, 
                obs[..., 1, :], 
                None, 
                0, 
                None
            )
            action2 = act_dist2.sample()
            action_list.append(jnp.stack([action1, action2], -1))




def create_loss(config, model, name='zero'):
    loss = Loss(config=config, model=model, name=name)

    return loss


def compute_actor_loss(
    config, 
    data, 
    stats, 
    act_dist, 
):
    if not config.get('policy_life_mask', True):
        sample_mask = data.sample_mask
    else:
        sample_mask = None

    if config.pg_type == 'pg':
        raw_pg_loss = jax_loss.pg_loss(
            advantage=stats.advantage, 
            logprob=stats.pi_logprob, 
            ratio=stats.ratio, 
        )
    elif config.pg_type == 'ppo':
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
        n=data.n
    )
    stats.raw_pg_loss = raw_pg_loss
    stats.scaled_pg_loss = scaled_pg_loss
    stats.pg_loss = pg_loss

    entropy = act_dist.entropy()
    scaled_entropy_loss, entropy_loss = jax_loss.entropy_loss(
        entropy_coef=stats.entropy_coef, 
        entropy=entropy, 
        mask=sample_mask, 
        n=data.n
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
    if config.get('value_life_mask', False):
        sample_mask = data.sample_mask
    else:
        sample_mask = None

    value_loss_type = config.value_loss
    if value_loss_type == 'huber':
        raw_value_loss = jax_loss.huber_loss(
            stats.value, 
            stats.v_target, 
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
            n=data.n,
        )
    else:
        raise ValueError(f'Unknown value loss type: {value_loss_type}')
    new_stats.raw_v_loss = raw_value_loss
    scaled_value_loss, value_loss = jax_loss.to_loss(
        raw_value_loss, 
        coef=stats.value_coef, 
        mask=sample_mask, 
        n=data.n
    )
    
    new_stats.scaled_v_loss = scaled_value_loss
    new_stats.v_loss = value_loss

    return value_loss, new_stats


def compute_model_loss(
    config, 
    data, 
    stats
):
    if config.model_loss_type == 'mbpo':
        mean_loss, var_loss = jax_loss.mbpo_model_loss(
            stats.model_mean, 
            stats.model_logvar, 
            data.next_obs
        )

        mean_loss = jnp.mean(mean_loss, [0, 1, 2])
        var_loss = jnp.mean(var_loss, [0, 1, 2])
        chex.assert_rank([mean_loss, var_loss], 1)
    else:
        raise NotImplementedError

    return mean_loss, var_loss


def record_target_adv(stats):
    stats.explained_variance = jax_math.explained_variance(
        stats.v_target, stats.value)
    stats.v_target_unit_std = jnp.std(stats.v_target, axis=-1)
    stats.raw_adv_unit_std = jnp.std(stats.raw_adv, axis=-1)
    return stats


def record_policy_stats(data, stats, act_dist):
    stats.diff_frac = jax_math.mask_mean(
        lax.abs(stats.pi_logprob - data.mu_logprob) > 1e-5, 
        data.sample_mask, data.n)
    stats.approx_kl = .5 * jax_math.mask_mean(
        (stats.log_ratio)**2, data.sample_mask, data.n)
    stats.approx_kl_max = jnp.max(.5 * (stats.log_ratio)**2)
    if data.mu is not None:
        stats.pi = nn.softmax(act_dist.logits)
        stats.diff_pi = stats.pi - data.mu
    elif data.mu_mean is not None:
        stats.pi_mean = act_dist.mu
        stats.diff_pi_mean = act_dist.mu - data.mu_mean
        stats.pi_std = act_dist.std
        stats.diff_pi_std = act_dist.std - data.mu_std

    return stats
