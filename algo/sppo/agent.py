import numpy as np
import tensorflow as tf

from core.tf_config import build
from core.decorator import agent_config
from utility.tf_utils import explained_variance
from algo.ppo.base import PPOBase
from algo.ppo.loss import compute_ppo_loss


class Agent(PPOBase):
    @agent_config
    def __init__(self, *, dataset, env):
        super().__init__(dataset=dataset, env=env)

        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=(env.action_shape, env.action_dtype, 'action'),
            traj_ret=((), tf.float32, 'traj_ret'),
            value=((), tf.float32, 'value'),
            advantage=((), tf.float32, 'advantage'),
            logpi=((), tf.float32, 'logpi'),
        )
        self.learn = build(self._learn, TensorSpecs)

    def __call__(self, obs, evaluation=False, **kwargs):
        if obs.ndim % 2 != 0:
            obs = np.expand_dims(obs, 0)
        obs = self.normalize_obs(obs)
        
        return tf.nest.map_structure(
            lambda x: x.numpy(), self.model.action(obs, evaluation))
        
    @tf.function
    def _learn(self, obs, action, traj_ret, value, advantage, logpi, mask=None, state=None):
        old_value = value
        terms = {}
        with tf.GradientTape() as tape:
            x = self.encoder(obs)
            if state is not None:
                self.rnn(x, state, mask=mask)
            act_dist = self.actor(x)
            new_logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            # policy loss
            log_ratio = new_logpi - logpi
            ppo_loss, entropy, kl, p_clip_frac = compute_ppo_loss(
                log_ratio, advantage, self._clip_range, entropy)
            # value loss
            value = self.value(x)
            value_loss = self.compute_value_loss(value, traj_ret, old_value, terms)
            actor_loss = (ppo_loss - self._entropy_coef * entropy)
            value_loss = self._value_coef * value_loss
            ac_loss = actor_loss + value_loss

        terms['ac_norm'] = self._optimizer(tape, ac_loss)
        terms.update(dict(
            value=value,
            advantage=advantage, 
            ratio=tf.exp(log_ratio), 
            entropy=entropy, 
            kl=kl, 
            p_clip_frac=p_clip_frac,
            ppo_loss=ppo_loss,
            actor_loss=actor_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value)
        ))

        return terms