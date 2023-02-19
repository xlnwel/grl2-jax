from jax import lax, random
import jax.numpy as jnp

from core.elements.loss import LossBase
from core.typing import dict2AttrDict
from .utils import *


class Loss(LossBase):
    def q_loss(
        self, 
        theta, 
        rng, 
        policy_params, 
        target_qs_params, 
        temp_params, 
        data, 
        name='train/q', 
    ):
        rngs = random.split(rng, 3)
        stats = dict2AttrDict(self.config.stats, to_copy=True)
        assert len(policy_params) == len(self.model.aid2uids), (len(policy_params), len(self.model.aid2uids))

        next_data = dict2AttrDict({
            'obs': data.next_obs, 
            'state_reset': data.next_state_reset, 
            'state': data.next_state, 
            'action_mask': data.next_action_mask, 
        })
        next_action, next_logprob, _ = compute_joint_action_logprob(
            self.model, policy_params, rngs[0], next_data, bptt=self.config.prnn_bptt
        )

        q_data = data.slice((slice(None), slice(None), slice(1)))
        next_qs_state = None if q_data.next_state is None else q_data.next_state.qs
        next_q = compute_qs(
            self.modules.Q, 
            target_qs_params, 
            rngs[1], 
            q_data.next_global_state, 
            next_action, 
            q_data.next_state_reset, 
            next_qs_state, 
            bptt=self.config.qrnn_bptt, 
            return_minimum=True
        )
        _, temp = self.modules.temp(temp_params, rngs[2])
        q_target = compute_target(
            q_data.reward, 
            q_data.discount, 
            stats.gamma, 
            next_q, 
            temp, 
            next_logprob
        )
        q_target = lax.stop_gradient(q_target)

        qs_state = None if q_data.state is None else q_data.state.qs
        action = data.action.reshape(*data.action.shape[:2], 1, -1)
        qs = compute_qs(
            self.modules.Q, 
            theta, 
            rngs[3], 
            q_data.global_state, 
            action, 
            q_data.state_reset, 
            qs_state, 
            bptt=self.config.qrnn_bptt
        )
        loss, stats = compute_q_loss(
            self.config, qs, q_target, data, stats
        )

        return loss, stats

    def policy_loss(
        self, 
        theta, 
        rng, 
        qs_params, 
        temp_params, 
        data, 
        stats, 
        name='train/policy', 
    ):
        if not stats:
            stats = dict2AttrDict(self.config.stats, to_copy=True)
        rngs = random.split(rng, 3)

        action, logprob, act_dists = compute_joint_action_logprob(
            self.model, theta, rngs[0], data, bptt=self.config.prnn_bptt
        )
        stats.logprob = logprob
        stats.update(act_dists[-1].get_stats('pi'))
        stats.entropy = act_dists[-1].entropy()

        q_data = data.slice((slice(None), slice(None), slice(1)))
        qs_state = None if q_data.state is None else q_data.state.qs
        q = compute_qs(
            self.modules.Q, 
            qs_params, 
            rngs[1], 
            q_data.global_state, 
            action, 
            q_data.state_reset, 
            qs_state, 
            bptt=self.config.qrnn_bptt, 
            return_minimum=True
        )
        stats.q = q
        _, temp = self.modules.temp(temp_params, rngs[2])
        loss, stats = compute_policy_loss(
            self.config, q, logprob, temp, data, stats
        )

        return loss, stats

    def temp_loss(
        self, 
        theta, 
        rng, 
        stats, 
    ):
        log_temp, temp = self.modules.temp(theta, rng)
        target_entropy = self.config.get(
            'target_entropy', -self.model.config.policy.action_dim)
        stats.target_entropy = target_entropy
        stats.temp = temp
        raw_temp_loss = - log_temp * (stats.logprob + target_entropy)
        stats.scaled_temp_loss, loss = jax_loss.to_loss(
            raw_temp_loss, 
            coef=stats.temp_coef, 
        )
        stats.temp_loss = loss

        return loss, stats


def create_loss(config, model, name='masac'):
    loss = Loss(config=config, model=model, name=name)

    return loss
