import logging
import tensorflow as tf

from utility.typing import EnvOutput
from core.agent import AgentBase
from core.decorator import override


logger = logging.getLogger(__name__)


def collect(buffer, env, env_step, reset, next_obs, **kwargs):
    buffer.add(**kwargs)


def get_data_format(*, env, **kwargs):
    obs_dtype = tf.uint8 if len(env.obs_shape) == 3 else tf.float32
    action_dtype = tf.int32 if env.is_action_discrete else tf.float32
    data_format = dict(
        obs=((None, *env.obs_shape), obs_dtype),
        action=((None, *env.action_shape), action_dtype),
        value=((None, ), tf.float32), 
        traj_ret=((None, ), tf.float32),
        advantage=((None, ), tf.float32),
        logpi=((None, ), tf.float32),
    )

    return data_format


class Agent(AgentBase):
    """ Initialization """
    @override(AgentBase)
    def _post_init(self, env, dataset):
        super()._post_init(env, dataset)

        self._huber_threshold = getattr(self, '_huber_threshold', None)

        self._last_obs = None   # we record last obs before training to compute the last value
        self._value_update = getattr(self, '_value_update', None)

    """ Standard PPO methods """
    def before_run(self, env):
        pass

    def record_last_env_output(self, env_output):
        self._env_output = EnvOutput(
            self.model.normalize_obs(env_output.obs), env_output.reward,
            env_output.discount, env_output.reset)

    def compute_value(self, obs=None):
        # be sure you normalize obs first if obs normalization is required
        obs = self._env_output.obs if obs is None else obs
        return self.model.compute_value(obs).numpy()

    def _store_additional_stats(self):
        self.store(**self.model.get_rms_stats())
        self.store(**self.dataset.compute_mean_max_std('reward'))
        # self.store(**self.dataset.compute_mean_max_std('obs'))
        self.store(**self.dataset.compute_mean_max_std('advantage'))
        self.store(**self.dataset.compute_mean_max_std('value'))
        self.store(**self.dataset.compute_mean_max_std('traj_ret'))

    def _sample_learn(self):
        for i in range(self.N_EPOCHS):
            for j in range(1, self.N_MBS+1):
                with self._sample_timer:
                    data = self._sample_data()

                data = {k: tf.convert_to_tensor(v) for k, v in data.items()}

                with self._learn_timer:
                    terms = self.trainer.learn(**data)

                kl = terms.pop('kl').numpy()
                value = terms.pop('value').numpy()
                terms = {f'train/{k}': v.numpy() for k, v in terms.items()}

                self.store(**terms)
                if getattr(self, '_max_kl', None) and kl > self._max_kl:
                    break
                if self._value_update == 'reuse':
                    self.dataset.update('value', value)

                self._after_train_step()

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
            
            self._after_train_epoch()

        n = i * self.N_MBS + j
        self.store(**{
            'stats/policy_updates': n,
            'train/kl': kl,
            'train/value': value,
            'time/sample_mean': self._sample_timer.average(),
            'time/learn_mean': self._learn_timer.average(),
        })

        if self._to_summary(self.train_step + n):
            self._summary(data, terms)

        return n
    
    def _sample_data(self):
        return self.dataset.sample()

    def _after_train_step(self):
        """ Does something after each training step """
        pass

    def _after_train_epoch(self):
        """ Does something after each training epoch """
        pass

    def _store_buffer_stats(self):
        self.store(**self.dataset.compute_mean_max_std('reward'))
        # self.store(**self.dataset.compute_mean_max_std('obs'))
        self.store(**self.dataset.compute_mean_max_std('advantage'))
        self.store(**self.dataset.compute_mean_max_std('value'))
        self.store(**self.dataset.compute_mean_max_std('traj_ret'))
