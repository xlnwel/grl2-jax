import tensorflow as tf

from utility.tf_utils import explained_variance
from utility.rl_loss import ppo_loss, quantile_regression_loss
from utility.schedule import TFPiecewiseSchedule
from core.optimizer import Optimizer
from core.tf_config import build
from core.decorator import override
from algo.ppo.base import PPOBase


class Agent(PPOBase):
    """ Initialization """
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)
        self._batch_size = env.n_envs * self.N_STEPS // self.N_MBS

    @override(PPOBase)
    def _construct_optimizers(self):
        if getattr(self, 'schedule_lr', False):
            assert isinstance(self._lr, list), self._lr
            self._lr = TFPiecewiseSchedule(self._lr)
        actor_models = [self.encoder, self.actor]
        self._actor_opt = Optimizer(
            self._optimizer, actor_models, self._lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)
        value_models = [self.encoder, self.quantile, self.value]
        self._value_opt = Optimizer(
            self._optimizer, value_models, self._lr, 
            clip_norm=self._clip_norm, scales=[1/64, 1, 1],
            epsilon=self._opt_eps)

    @override(PPOBase)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=((env.action_dim,), env.action_dtype, 'action'),
            value=((self.N,), tf.float32, 'value'),
            traj_ret=((self.N,), tf.float32, 'traj_ret'),
            advantage=((self.N,), tf.float32, 'advantage'),
            logpi=((), tf.float32, 'logpi'),
        )
        self.learn = build(self._learn, TensorSpecs, batch_size=self._batch_size)

    # @override(PPOBase)
    def _summary(self, data, terms):
        tf.summary.histogram('sum/value', data['value'], step=self._env_step)
        tf.summary.histogram('sum/logpi', data['logpi'], step=self._env_step)

    def _process_input(self, env_output, evaluation):
        obs, kwargs = super()._process_input(env_output, evaluation)
        kwargs['tau_hat'] = self._tau_hat
        return obs, kwargs

    def before_run(self, env):
        self._tau_hat = self.quantile.sample_tau(env.n_envs)

    def compute_value(self, obs=None):
        # be sure you normalize obs first if obs normalization is required
        obs = obs or self._last_obs
        return self.model.compute_value(obs, self._tau_hat).numpy()

    @tf.function
    def _learn(self, obs, action, value, traj_ret, advantage, logpi, state=None, mask=None, additional_input=[]):
        old_value = value
        terms = {}
        advantage = tf.reduce_mean(advantage, axis=-1)
        with tf.GradientTape(persistent=True) as tape:
            x = self.encoder(obs)
            if state is not None:
                x, _ = self.rnn(x, state, mask=mask, additional_input=additional_input)
            act_dist = self.actor(x)
            new_logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            # policy loss
            log_ratio = new_logpi - logpi
            policy_loss, entropy, kl, p_clip_frac = ppo_loss(
                log_ratio, advantage, self._clip_range, entropy)
            # value loss
            tau_hat, qt_embed = self.quantile(x, self.N)
            x = tf.expand_dims(x, 1)
            value = self.value(x, qt_embed)
            value_ext = tf.expand_dims(value, axis=-1)
            traj_ret_ext = tf.expand_dims(traj_ret, axis=1)
            value_loss = quantile_regression_loss(
                value_ext, traj_ret_ext, tau_hat, kappa=self.KAPPA)
            value_loss = tf.reduce_mean(value_loss)
            
            tf.debugging.assert_shapes([
                [policy_loss, (None, )],
                [value_loss, (None, )],
            ])
            
            actor_loss = (policy_loss - self._entropy_coef * entropy)
            value_loss = self._value_coef * value_loss

        terms['actor_norm'] = self._actor_opt(tape, actor_loss)
        terms['value_norm'] = self._value_opt(tape, value_loss)
        terms.update(dict(
            value=tf.reduce_mean(value),
            traj_ret=tf.reduce_mean(traj_ret), 
            advantage=tf.reduce_mean(advantage), 
            ratio=tf.exp(log_ratio), 
            entropy=entropy, 
            kl=kl, 
            p_clip_frac=p_clip_frac,
            ppo_loss=policy_loss,
            actor_loss=actor_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value)
        ))

        return terms
