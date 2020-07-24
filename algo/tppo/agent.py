import cloudpickle
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.display import pwc
from utility.schedule import TFPiecewiseSchedule
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config, step_track
from core.optimizer import Optimizer
from algo.ppo.base import PPOBase
from algo.tppo.loss import compute_tppo_loss, compute_value_loss


class Agent(PPOBase):
    @agent_config
    def __init__(self, dataset, env):
        super().__init__(dataset=dataset, env=env)

        # optimizer
        self._optimizer = Optimizer(
            self._optimizer, self.ac, self._lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)

        self._is_action_discrete = env.is_action_discrete

        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=(env.action_shape, env.action_dtype, 'action'),
            traj_ret=((), tf.float32, 'traj_ret'),
            value=((), tf.float32, 'value'),
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

    def reset_states(self, states=None):
        pass

    def get_states(self):
        return None

    def __call__(self, obs, deterministic=False, update_rms=False, **kwargs):
        if obs.ndim % 2 != 0:
            obs = np.expand_dims(obs, 0)
        if update_rms:
            self.update_obs_rms(obs)
        obs = self.normalize_obs(obs)
        if deterministic:
            return self.model.action(obs, deterministic).numpy()
        else:
            out = self.model.action(obs, deterministic)
            action, terms = tf.nest.map_structure(lambda x: x.numpy(), out)
            terms['obs'] = obs  # return normalized obs
            return action, terms

    @step_track
    def learn_log(self, step):
        for i in range(self.N_UPDATES):
            for j in range(self.N_MBS):
                data = self.dataset.sample()
                value = data['value']
                data = {k: tf.convert_to_tensor(v) for k, v in data.items()}
                terms = self.learn(**data)

                terms = {k: v.numpy() for k, v in terms.items()}

                terms['value'] = np.mean(value)
                kl, p_clip_frac, v_clip_frac = \
                    terms['kl'], terms['p_clip_frac'], terms['v_clip_frac']
                for k in ['kl', 'p_clip_frac', 'v_clip_frac']:
                    del terms[k]

                self.store(**terms)
                if getattr(self, '_max_kl', None) and kl > self._max_kl:
                    break
            if getattr(self, '_max_kl', None) and kl > self._max_kl:
                pwc(f'{self._model_name}: Eearly stopping after '
                    f'{i*self.N_MBS+j+1} update(s) due to reaching max kl.',
                    f'Current kl={kl:.3g}', color='blue')
                break
        self.store(kl=kl, p_clip_frac=p_clip_frac, v_clip_frac=v_clip_frac)
        if not isinstance(self._lr, float):
            step = tf.cast(self._env_step, tf.float32)
            self.store(lr=self._lr(step))
        return i * self.N_MBS + j + 1

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
            ppo_loss, entropy, p_clip_frac = compute_tppo_loss(
                log_ratio, kl, advantage, self._kl_weight, self._clip_range, entropy)
            # value loss
            value_loss, v_clip_frac = compute_value_loss(
                value, traj_ret, old_value, self._clip_range)

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
