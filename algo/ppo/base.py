import logging
import numpy as np
import tensorflow as tf

from utility.utils import Every
from utility.tf_utils import explained_variance
from utility.schedule import TFPiecewiseSchedule
from core.base import RMSBaseAgent
from core.decorator import step_track
from core.optimizer import Optimizer
from algo.ppo.loss import compute_ppo_loss, compute_value_loss


logger = logging.getLogger(__name__)

class PPOBase(RMSBaseAgent):
    def __init__(self, *, env, dataset):
        super().__init__()
        self.dataset = dataset
        self._construct_optimizers()
        self._to_summary = Every(self.LOG_PERIOD, self.LOG_PERIOD)
        self._last_obs = None
        logger.info(f'Value update scheme: {self._value_update}')

    def _construct_optimizers(self):
        if getattr(self, 'schedule_lr', False):
            assert isinstance(self._lr, list)
            self._lr = TFPiecewiseSchedule(self._lr)
        models = [self.encoder, self.actor, self.value]
        if hasattr(self, 'rnn'):
            models.append(self.rnn)
        self._optimizer = Optimizer(
            self._optimizer, models, self._lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)

    """ Standard PPO functions """
    def reset_states(self, state=None):
        pass

    def get_states(self):
        return None
    
    def record_last_obs(self, obs):
        self._last_obs = self.normalize_obs(obs)
        
    def compute_value(self, obs=None):
        # be sure you normalize obs first if normalization is required
        obs = obs or self._last_obs
        return self.model.compute_value(self._last_obs).numpy()

    @step_track
    def learn_log(self, step):
        for i in range(self.N_EPOCHS):
            for j in range(1, self.N_MBS+1):
                data = self.dataset.sample()
                data = {k: tf.convert_to_tensor(v) for k, v in data.items()}
                terms = self.learn(**data)

                terms = {f'train/{k}': v.numpy() for k, v in terms.items()}
                kl = terms.pop('train/kl')
                value = terms.pop('train/value')
                self.store(**terms, value=value.mean())
                if getattr(self, '_max_kl', None) and kl > self._max_kl:
                    break
                if self._value_update == 'reuse':
                    self.dataset.update('value', value)
            if getattr(self, '_max_kl', None) and kl > self._max_kl:
                logger.info(f'{self._model_name}: Eearly stopping after '
                    f'{i*self.N_MBS+j} update(s) due to reaching max kl.'
                    f'Current kl={kl:.3g}')
                break
            
            if self._value_update == 'once':
                self.dataset.update_value_with_func(self.compute_value)
            if self._value_update is not None:
                last_value = self.compute_value()
                self.dataset.finish(last_value)
        self.store(**{'aux/kl': kl})
        # if not isinstance(self._lr, float):
        #     step = tf.cast(self._env_step, tf.float32)
        #     self.store(lr=self._lr(step))
        
        if self._to_summary(step):
            self.summary(data, terms)

        return i * self.N_MBS + j

    def summary(self, data, terms):
        pass
        # tf.summary.histogram('traj_ret', data['train/traj_ret'], step=self._env_step)

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

    def compute_value_loss(self, value, traj_ret, old_value, terms):
        value_loss_type = getattr(self, '_value_loss', 'mse')
        if value_loss_type == 'mse':
            value_loss = .5 * tf.reduce_mean((value - traj_ret)**2)
        elif value_loss_type == 'clip':
            value_loss, v_clip_frac = compute_value_loss(
                value, traj_ret, old_value, self._clip_range)
            terms['v_clip_frac'] = v_clip_frac
        else:
            raise ValueError(f'Unknown value loss type: {value_loss_type}')
        return value_loss