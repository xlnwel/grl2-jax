import logging
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from tools.schedule import TFPiecewiseSchedule
from tools.tf_utils import explained_variance, tensor2numpy
from jax_utils.jax_loss import ppo_loss
from optimizers.adam import Adam
from core.tf_config import build
from core.optimizer import Optimizer
from core.decorator import override
from algo.ppo.base import PPOBase, collect


logger = logging.getLogger(__name__)

class Agent(PPOBase):
    """ Initialization """
    @override(PPOBase)
    def _add_attributes(self, env, dateset):
        super()._add_attributes(env, dateset)
        assert self.N_SEGS <= self.N_PI, f'{self.N_SEGS} > {self.N_PI}'
        self.N_AUX_MBS = self.N_SEGS * self.N_AUX_MBS_PER_SEG
        self._aux_batch_size = env.n_envs * self.N_STEPS // self.N_AUX_MBS_PER_SEG

    @override(PPOBase)
    def _construct_optimizers(self):
        if getattr(self, 'schedule_lr', False):
            assert isinstance(self._actor_lr, list)
            assert isinstance(self._value_lr, list)
            assert isinstance(self._aux_lr, list)
            self._actor_lr = TFPiecewiseSchedule(self._actor_lr)
            self._value_lr = TFPiecewiseSchedule(self._value_lr)
            self._aux_lr = TFPiecewiseSchedule(self._aux_lr)

        actor_models = [self.encoder, self.actor, self.aux_value, self.aux_advantage]
        if hasattr(self, 'rnn'):
            actor_models.append(self.rnn)
        self._actor_opt = Optimizer(
            Adam, actor_models, self._actor_lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)

        value_models = [self.value]
        if hasattr(self, 'value_encoder'):
            value_models.append(self.value_encoder)
        if hasattr(self, 'value_rnn'):
            value_models.append(self.value_rnn)
        self._value_opt = Optimizer(
            self._optimizer, value_models, self._value_lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)
        
        # aux_models = list(self.model.values())
        self._aux_opt = Optimizer(
            self._optimizer, value_models, self._aux_lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)

        self._meta_opt = Optimizer(
            Adam, self.meta, self._meta_lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)

    @override(PPOBase)
    def _build_train(self, env):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=(env.action_shape, env.action_dtype, 'action'),
            value=((), tf.float32, 'value'),
            traj_ret=((), tf.float32, 'traj_ret'),
            advantage=((), tf.float32, 'advantage'),
            logpi=((), tf.float32, 'logpi'),
        )
        self.train = build(self._learn, TensorSpecs, batch_size=self._batch_size)
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            logits=((env.action_dim,), tf.float32, 'logits'),
            value=((), tf.float32, 'value'),
            traj_ret=((), tf.float32, 'traj_ret'),
        )
        self.aux_learn = build(self._aux_learn, TensorSpecs, batch_size=self._aux_batch_size)
        self.value_learn = build(self._value_learn, TensorSpecs, batch_size=self._aux_batch_size)

    """ PPG methods """
    def compute_aux_data(self, obs):
        out = self.model.compute_aux_data(obs)
        out = tensor2numpy(out)
        return out

    def aux_train_log(self, step):
        for i in range(self.N_AUX_EPOCHS):
            for j in range(1, self.N_AUX_MBS+1):
                data = self.dataset.sample_aux_data()
                data = {k: tf.convert_to_tensor(v) for k, v in data.items()}
                terms = self.value_learn(**data)

                terms = {f'aux/{k}': v.numpy() for k, v in terms.items()}
                # kl = terms.pop('aux/kl')
                value = terms.pop('aux/value')
                self.store(**terms, value=value.mean())
                if self._value_update == 'reuse':
                    self.dataset.aux_update('value', value)
            if self._value_update == 'once':
                self.dataset.compute_aux_data_with_func(self.compute_value)
            if self._value_update is not None:
                last_value = self.compute_value()
                self.dataset.aux_finish(last_value)
        if self.N_AUX_EPOCHS:
            # self.store(**{'aux/kl': kl})
            
            if self._to_summary(step):
                self._summary(data, terms)

            return i * self.N_MBS + j
        else:
            return 0

    @tf.function
    def _learn(self, obs, action, value, traj_ret, advantage, logpi, 
                    state=None, mask=None):
        with tf.GradientTape(persistent=True) as tape:
            terms = self._learn_impl(obs, action, value, traj_ret, advantage, logpi, 
                                state, mask)
            x = self.encoder(obs)
            if state is not None:
                x, state = self.rnn(x, state, mask=mask)
            act_dist = self.actor(x)
            new_logpi = act_dist.log_prob(action)
            log_ratio = new_logpi - logpi
            ratio = tf.exp(log_ratio)
            meta_loss = -tf.reduce_mean(advantage * ratio)
        var_list = self.encoder.trainable_variables \
                + self.actor.trainable_variables
                # + self.aux_value.trainable_variables \
                # + self.aux_advantage.trainable_variables
        out_grads = tape.gradient(meta_loss, var_list)
        out_grads = tape.gradient(
            self._actor_opt.get_transformed_grads(var_list), 
            self._actor_opt.grads, 
            output_gradients=out_grads)
        terms['meta_norm'] = self._meta_opt(
            tape, 
            self._actor_opt.grads, 
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

    def _learn_impl(self, obs, action, value, traj_ret, advantage, logpi, state=None, mask=None):
        old_value = value
        terms = {}
        with tf.GradientTape() as tape:
            x = self.encoder(obs)
            if state is not None:
                x, state = self.rnn(x, state, mask=mask)    
            act_dist = self.actor(x)
            new_logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            log_ratio = new_logpi - logpi
            policy_loss, entropy, kl, p_clip_frac = ppo_loss(
                log_ratio, advantage, self._clip_range, entropy)
            entropy_coef = self.model.get_meta('entropy_coef')
            actor_loss = policy_loss - entropy_coef * entropy
        
            value = self.aux_value(x)
            value_loss, v_clip_frac = self._compute_value_loss(
                value, traj_ret, old_value)

            one_hot = tf.one_hot(action, self._action_dim)
            x_a = tf.concat([x, one_hot], axis=-1)
            adv = self.aux_advantage(x_a)
            tf.debugging.assert_rank(adv, 1)
            adv_loss = .5 * tf.reduce_mean((adv - advantage)**2)
            value_coef = self.model.get_meta('value_coef')
            adv_coef = self.model.get_meta('adv_coef')
            loss = actor_loss + value_coef * value_loss \
                + adv_coef * adv_loss
        terms['actor_norm'] = self._actor_opt(tape, loss)

        if self.model.architecture == 'dual':
            with tf.GradientTape() as tape:
                x_value = self.value_encoder(obs)
                value = self.value(x_value)
                value_loss, v_clip_frac = self._compute_value_loss(
                    value, traj_ret, old_value)
            terms['value_norm'] = self._value_opt(tape, value_loss)

        terms.update(dict(
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            adv_coef=adv_coef,
            x=tf.reduce_mean(x),
            x_std=tf.math.reduce_std(x),
            x_max=tf.math.reduce_max(x),
            x_value=tf.reduce_mean(x_value),
            x_value_std=tf.math.reduce_std(x_value),
            x_value_max=tf.math.reduce_max(x_value),
            value=value,
            advantage=advantage, 
            ratio=tf.exp(log_ratio), 
            entropy=entropy, 
            kl=kl, 
            p_clip_frac=p_clip_frac,
            ppo_loss=policy_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value),
            v_clip_frac=v_clip_frac
        ))

        return terms

    @tf.function
    def _aux_learn(self, obs, logits, value, traj_ret, mask=None, state=None):
        old_value = value
        terms = {}
        with tf.GradientTape() as tape:
            x = self.encoder(obs)
            if state is not None:
                x, state = self.rnn(x, state, mask=mask)
            act_dist = self.actor(x)
            old_dist = tfd.Categorical(logits=logits)
            kl = tf.reduce_mean(old_dist.kl_divergence(act_dist))
            actor_loss = bc_loss = self._bc_coef * kl
            if hasattr(self, 'value_encoder'):
                aux_value = self.aux_value(x)
                aux_loss = .5 * tf.reduce_mean((aux_value - traj_ret)**2)
                terms['bc_loss'] = bc_loss
                terms['aux_loss'] = aux_loss
                actor_loss = aux_loss + bc_loss

                x_value = self.value_encoder(obs)
                value = self.value(x_value)
            else:
                # allow gradients from value head if using a shared encoder
                value = self.value(x)

            value_loss, v_clip_frac = self._compute_value_loss(
                value, traj_ret, old_value)
            loss = actor_loss + value_loss

        terms['actor_norm'] = self._aux_opt(tape, loss)

        terms.update(dict(
            value=value, 
            kl=kl, 
            actor_loss=actor_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value),
            v_clip_frac=v_clip_frac
        ))

        return terms

    @tf.function
    def _value_learn(self, obs, logits, value, traj_ret, state=None, mask=None):
        old_value = value
        terms = {}
        with tf.GradientTape() as tape:
            x = self.value_encoder(obs)
            if state is not None:
                x, state = self.value_rnn(x, state, mask=mask)
            value = self.value(x)
            value_loss, v_clip_frac = self._compute_value_loss(
                value, traj_ret, old_value)
        value_norm = self._value_opt(tape, value_loss)
        terms.update(dict(
            value=value,
            value_norm=value_norm,
            value_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value),
            v_clip_frac=v_clip_frac,
        ))
        return terms
