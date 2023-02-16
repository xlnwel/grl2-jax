from jax import lax, nn, random
import jax.numpy as jnp

import sys, os
sys.path.append(os.path.dirname(os.path.realpath('__file__')))

from core.elements.loss import LossBase
from core.typing import dict2AttrDict, AttrDict
from jax_tools import jax_loss, jax_utils
from tools.rms import denormalize
from algo.ppo.elements.utils import *
from jax_tools.jax_div import kl_from_distributions


class Loss(LossBase):
    def model_loss(
        self,
        theta,
        rng,
        data,
        name='model'
    ):
        obs, _ = jax_utils.split_data(data.obs, data.next_obs, axis=1)
        global_state, _ = jax_utils.split_data(data.global_state, data.next_global_state, axis=1)
        # TODO: whether split state
        rngs = random.split(rng, 8)

        # ========== following is to compute RSSM model loss ==========

        # compute observation reconstruction loss
        # TODO: consider the reconstruction for state
        # obs_embed = self.modules.encoder(theta.encoder rngs[0], obs)
        obs_embed = self.modules.obs_encoder(theta.obs_encoder, rngs[0], obs)
        state_embed = self.modules.state_encoder(theta.state_encoder, rngs[1], global_state)
        
        # recons_obs = self.modules.decoder(theta.decoder, rngs[1], obs_embed)
        # recons_loss, stats = compute_reconstruction_loss(
            # self.config, obs, recons_obs, stats)
        # compute rssm loss
        obs_post, _ = self.modules.rssm(theta.rssm, rngs[2], data.action, obs_embed, data.reset, imagine=False)
        state_post, state_prior = self.modules.rssm(theta.rssm, rngs[3], data.action, state_embed, data.reset, imagine=False)
        rssm_loss, stats = compute_rssm_loss(
            self.config, state_post, state_prior, stats
        )
        # fetch the feature at each timestep
        state_rssm_feat = self.modules.rssm_feat(state_post)
        obs_rssm_feat = self.modules.rssm_feat(obs_post)
        next_rssm_feat = state_rssm_feat[:, 1:]
        reward_dist = self.modules.reward(
            theta.reward, rngs[4], state_rssm_feat, data.action)
        reward_loss, stats = compute_reward_loss(
            self.config, reward_dist, data.reward, stats)
        discount_dist = self.modules.discount(
            theta.discount, rngs[5], next_rssm_feat)
        discount_loss, stats = compute_discount_loss(
            self.config, discount_dist, data.discount, stats)

        # compute reconstruction loss
        recons_state1 = self.modules.decoder(theta.decoder, rngs[6], state_rssm_feat)
        recons_state2 = self.modules.decoder(theta.decoder, rngs[7], obs_rssm_feat)
        recons_loss1, stats = compute_reconstruction_loss(
            self.config, global_state, recons_state1, stats
        )
        recons_loss2, stats = compute_reconstruction_loss(
            self.config, global_state, recons_state2, stats
        )

        # compute reg loss
        reg_loss, stats = compute_reg_loss(
            self.config, state_post, obs_post, stats
        )
        
        model_loss = rssm_loss + reward_loss + discount_loss + recons_loss1 + recons_loss2 + reg_loss
        stats.model_loss = model_loss
        
        return model_loss, stats

    def rl_loss(
        self,
        theta,
        rng,
        data,
        name='rl'
    ):
        rngs = random.split(rng, 2)
        # ========== following is to compute rl loss ==========

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
            data.state_reset, 
            None if data.state is None else data.state.value, 
            bptt=self.config.vrnn_bptt, 
            seq_axis=1, 
        )

        act_dist, stats.pi_logprob, stats.log_ratio, stats.ratio = \
            compute_policy(
                self.model, 
                theta.policy, 
                rngs[1], 
                data.obs, 
                data.next_obs, 
                data.action, 
                data.mu_logprob, 
                data.state_reset[:, :-1] if 'state_reset' in data else None, 
                None if data.state is None else data.state.policy, 
                action_mask=data.action_mask, 
                next_action_mask=data.next_action_mask, 
                bptt=self.config.prnn_bptt, 
                seq_axis=1, 
            )
        stats = record_policy_stats(data, stats, act_dist)

        if 'advantage' in data:
            stats.raw_adv = data.pop('advantage')
            stats.v_target = data.pop('v_target')
        else:
            if self.config.popart:
                value = lax.stop_gradient(denormalize(
                    stats.value, data.popart_mean, data.popart_std))
                next_value = denormalize(next_value, data.popart_mean, data.popart_std)
            else:
                value = lax.stop_gradient(stats.value)

            v_target, stats.raw_adv = jax_loss.compute_target_advantage(
                config=self.config, 
                reward=data.reward, 
                discount=data.discount, 
                reset=data.reset, 
                value=value, 
                next_value=next_value, 
                ratio=lax.stop_gradient(stats.ratio), 
                gamma=stats.gamma, 
                lam=stats.lam, 
                axis=1
            )
            stats.v_target = lax.stop_gradient(v_target)
        stats = record_target_adv(stats)

        stats.advantage = norm_adv(
            self.config, 
            stats.raw_adv, 
            sample_mask=data.sample_mask, 
            n=data.n, 
            epsilon=self.config.get('epsilon', 1e-5)
        )

        actor_loss, stats = compute_actor_loss(
            self.config, 
            data, 
            stats, 
            act_dist=act_dist, 
        )

        kl_stats = dict(
            logp=stats.pi_logprob, 
            logq=data.mu_logprob, 
            sample_prob=data.mu_logprob, 
            p_logits=stats.pi_logits, 
            q_logits=data.mu_logits, 
            p_loc=stats.pi_loc,  
            p_scale=stats.pi_scale, 
            q_loc=data.mu_loc,  
            q_scale=data.mu_scale, 
            action_mask=data.action_mask, 
            sample_mask=data.sample_mask, 
            n=data.n
        )
        stats.kl, stats.raw_kl_loss, stats.kl_loss = jax_loss.compute_kl(
            kl_type=self.config.kl_type, 
            kl_coef=self.config.kl_coef, 
            **kl_stats
        )
        value_loss, stats = compute_vf_loss(
            self.config, 
            data, 
            stats, 
        )
        rl_loss = actor_loss + value_loss + stats.kl_loss
        stats.rl_loss = rl_loss
        
        return rl_loss, stats

"""
The function of `create_loss' is called by outside.
The functions of `create_model_loss` ... are called by Loss
"""

def create_loss(config, model, name='model'):
    loss = Loss(config=config, model=model, name=name)

    return loss

def compute_reg_loss(
    config, state_post, obs_post, stats
):
    loss = .5 * (state_post.deter - obs_post.deter) ** 2
    stats.reg_loss = jnp.mean(loss, [0, 1, 2, 4])
    reg_loss = jnp.sum(stats.reg_loss)
    stats.reg_loss = config.reg_coef * reg_loss
    return config.reg_coef * reg_loss, stats
    
def compute_reconstruction_loss(
    config, state, recons_state, stats
):
    assert getattr(config, "reconstruction_loss_type", "mse") == "mse"
    loss = .5 * (recons_state - state) ** 2
    stats.recons_loss = jnp.mean(loss, [0, 1, 2, 4])
    recons_loss = jnp.sum(stats.recons_loss)
    stats.recons_loss = config.recons_coef * recons_loss
    return config.recons_coef * recons_loss, stats

def compute_rssm_loss(
    config, post_state, prior_state, stats
):  
    # TODO: check the order of post and prior when computing kl
    rssm_loss = kl_from_distributions(
        p_loc=post_state.mean, q_loc=prior_state.mean,
        p_scale=post_state.std, q_scale=prior_state.std
    )
    stats.rssm_loss = jnp.mean(rssm_loss, [0, 1, 2, 4])
    rssm_loss = jnp.sum(stats.rssm_loss)
    stats.rssm_loss = config.rssm_coef * rssm_loss
    return config.rssm_coef * rssm_loss, stats

def compute_reward_loss(
    config, reward_dist, reward, stats
):
    pred_reward = reward_dist.mode()
    reward_loss = jnp.mean(.5 * (pred_reward - reward)**2)
    stats.pred_reward = pred_reward
    stats.reward_mae = lax.abs(pred_reward - reward)
    stats.reward_consistency = jnp.mean(stats.reward_mae < .1)
    stats.reward_loss = config.reward_coef * reward_loss

    return config.reward_coef * reward_loss, stats

def compute_discount_loss(
    config, discount_dist, discount, stats
):
    discount_loss = - discount_dist.log_prob(discount)
    discount_loss = jnp.mean(discount_loss)
    discount = discount_dist.mode()
    stats.pred_discount = discount
    stats.discount_consistency = jnp.mean(stats.discount == discount)
    stats.discount_loss = config.discount_coef * discount_loss

    return config.discount_coef * discount_loss, stats
    