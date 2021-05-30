import logging
import tensorflow as tf

from utility.tf_utils import explained_variance
from utility.rl_loss import ppo_loss
from utility.schedule import TFPiecewiseSchedule
from utility.adam import Adam
from core.optimizer import Optimizer
from core.tf_config import build
from core.decorator import override, step_track
from algo.ppo.base import PPOBase


logger = logging.getLogger(__name__)

class Agent(PPOBase):
    """ Initialization """
    @override(PPOBase)
    def _construct_optimizers(self):
        if getattr(self, 'schedule_lr', False):
            assert isinstance(self._lr, list), self._lr
            self._lr = TFPiecewiseSchedule(self._lr)
        # TODO: should we put meta_embed here?
        ac = [self.encoder, self.actor, self.value, self.meta_embed]
        self._ac_opt = Optimizer(
            Adam, ac, self._ac_lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)
        self._meta_opt = Optimizer(
            Adam, self.meta, self._meta_lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)

    @override(PPOBase)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=(env.action_shape, env.action_dtype, 'action'),
            value=((), tf.float32, 'value'),
            traj_ret=((), tf.float32, 'traj_ret'),
            advantage=((), tf.float32, 'advantage'),
            logpi=((), tf.float32, 'logpi'),
        )
        self.learn = build(self._learn, TensorSpecs)
        self.meta_learn = build(self._meta_learn, TensorSpecs)

    @step_track
    def learn_log(self, step):
        for i in range(self.N_EPOCHS):
            for j in range(1, self.N_MBS+1):
                with self._sample_timer:
                    data = self.dataset.sample()
                data = {k: tf.convert_to_tensor(v) for k, v in data.items()}
                
                with self._learn_timer:
                    terms = self.meta_learn(**data) if i == 0 else self.learn(**data)

                terms = {f'train/{k}': v.numpy() for k, v in terms.items()}
                kl = terms.pop('train/kl')
                value = terms.pop('train/value')
                self.store(**terms, **{'train/value': value.mean()})
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
        
        if self._to_summary(step):
            self._summary(data, terms)

        self.store(**{
            'train/kl': kl,
            'time/sample_mean': self._sample_timer.average(),
            'time/learn_mean': self._learn_timer.average()
        })

        _, rew_rms = self.get_running_stats()
        if rew_rms:
            self.store(**{
                'train/reward_rms_mean': rew_rms.mean,
                'train/reward_rms_var': rew_rms.var
            })

        return i * self.N_MBS + j

    # @override(PPOBase)
    # def _summary(self, data, terms):
    #     tf.summary.histogram('sum/value', data['value'], step=self._env_step)
    #     tf.summary.histogram('sum/logpi', data['logpi'], step=self._env_step)

    @tf.function
    def _meta_learn(self, obs, action, value, traj_ret, advantage, logpi, 
                    state=None, mask=None, additional_input=[]):
        terms = {}
        with tf.GradientTape(persistent=True) as meta_tape:
            terms = self._learn_impl(obs, action, value, traj_ret, advantage, logpi, 
                            state, mask, additional_input)
            act_dist, value = self._forward_model(obs)
            new_logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            # policy loss
            log_ratio = new_logpi - logpi
            ratio = tf.exp(log_ratio)
            meta_loss = -tf.reduce_mean(advantage * ratio) + self._value_coef * (value - traj_ret)**2
        var_list = self.encoder.trainable_variables \
                + self.actor.trainable_variables \
                + self.value.trainable_variables
        out_grads = meta_tape.gradient(meta_loss, var_list)
        out_grads = meta_tape.gradient(
            self._ac_opt.get_transformed_grads(var_list), 
            self._ac_opt.grads, 
            output_gradients=out_grads)
        terms['meta_norm'] = self._meta_opt(
            meta_tape, 
            self._ac_opt.grads, 
            output_gradients=out_grads)

        meta_params = self.meta.trainable_variables
        meta_grads = self._meta_opt.grads
        meta_trans_grads = self._meta_opt.get_transformed_grads(meta_params)
        meta_dict = {f'{mp.name}_grad': mg for mp, mg in zip(meta_params, meta_grads)}
        meta_trans_dict = {f'{mp.name}_trans_grad': mg for mp, mg in zip(meta_params, meta_trans_grads)}
        terms.update(dict(
            meta_loss=meta_loss,
            **meta_dict,
            **meta_trans_dict,
        ))

        return terms

    @tf.function
    def _learn(self, obs, action, value, traj_ret, advantage, logpi, state=None, mask=None, additional_input=[]):
        return self._learn_impl(
            obs, action, value, traj_ret, advantage, logpi, 
            state, mask, additional_input)
    
    def _learn_impl(self, obs, action, value, traj_ret, advantage, logpi, state=None, mask=None, additional_input=[]):
        old_value = value
        terms = {}
        with tf.GradientTape() as tape:
            act_dist, value = self._forward_model(obs)
            new_logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            # policy loss
            log_ratio = new_logpi - logpi
            policy_loss, entropy, kl, p_clip_frac = ppo_loss(
                log_ratio, advantage, self._clip_range, entropy)
            # value loss
            value_loss, v_clip_frac = self._compute_value_loss(value, traj_ret, old_value)
            entropy_coef = self.model.get_meta('entropy_coef')
            actor_loss = (policy_loss - entropy_coef * entropy)
            value_coef = self.model.get_meta('value_coef')
            value_loss = value_coef * value_loss
            ac_loss = actor_loss + value_loss
        terms['ac_norm'] = self._ac_opt(tape, ac_loss)

        terms.update(dict(
            value=value,
            traj_ret=tf.reduce_mean(traj_ret), 
            advantage=tf.reduce_mean(advantage), 
            ratio=tf.exp(log_ratio), 
            entropy=entropy, 
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            kl=kl, 
            p_clip_frac=p_clip_frac,
            ppo_loss=policy_loss,
            actor_loss=actor_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value),
            v_clip_frac=v_clip_frac
        ))
        return terms
    
    def _forward_model(self, obs):
        x = self.model.encode(obs)
        act_dist = self.actor(x)
        value = self.value(x)
        return act_dist, value
    