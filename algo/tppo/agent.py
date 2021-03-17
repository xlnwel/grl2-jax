import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.rl_loss import tppo_loss
from core.tf_config import build
from core.decorator import override
from core.optimizer import Optimizer
from algo.ppo.base import PPOBase


class Agent(PPOBase):
    """ Initialization """
    @override(PPOBase)
    def _build_learn(self, env):
        # optimizer
        self._optimizer = Optimizer(
            self._optimizer, self.ac, self._lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)

        self._is_action_discrete = env.is_action_discrete

        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=(env.action_shape, env.action_dtype, 'action'),
            value=((), tf.float32, 'value'),
            traj_ret=((), tf.float32, 'traj_ret'),
            advantage=((), tf.float32, 'advantage'),
            logpi=((), tf.float32, 'logpi'),
        )
        if self._is_action_discrete:
            TensorSpecs['prob_params'] = ((env.action_dim,), tf.float32, 'logits')
        else:
            raise NotImplementedError
            # TODO: This requires to treat std as a state without batch_size dimension
            # A convinient way to do so is to retrieve std at the start of each learning epoch
            TensorSpecs['prob_params'] = [
                (env.action_shape, tf.float32, 'mean'),
                (env.action_shape, tf.float32, 'std', ()),
            ]
        self.learn = build(self._learn, TensorSpecs)

    @tf.function
    def _learn(self, obs, action, traj_ret, value, advantage, logpi, prob_params):
        old_value = value
        if self._is_action_discrete:
            old_act_dist = tfd.Categorical(prob_params)
        else:
            old_act_dist = tfd.MultivariateNormalDiag(*prob_params)
        with tf.GradientTape() as tape:
            act_dist, terms = self.ac(obs, return_terms=True)
            value = terms['value']
            new_logpi = act_dist.log_prob(action)
            kl = old_act_dist.kl_divergence(act_dist)
            entropy = act_dist.entropy()
            # policy loss
            log_ratio = new_logpi - logpi
            ppo_loss, entropy, p_clip_frac = tppo_loss(
                log_ratio, kl, advantage, self._kl_weight, self._clip_range, entropy)
            # value loss
            value_loss, v_clip_frac = self._compute_value_loss(value, traj_ret, old_value)

            policy_loss = (ppo_loss - self._entropy_coef * entropy)
            value_loss = self._value_coef * value_loss
            ac_loss = policy_loss + value_loss

        terms = dict(
            advantage=advantage, 
            ratio=tf.exp(log_ratio), 
            entropy=entropy, 
            kl=tf.reduce_mean(kl),
            p_clip_frac=p_clip_frac,
            v_clip_frac=v_clip_frac,
            ppo_loss=ppo_loss,
            v_loss=value_loss
        )
        terms['ac_norm'] = self._optimizer(tape, ac_loss)

        return terms
