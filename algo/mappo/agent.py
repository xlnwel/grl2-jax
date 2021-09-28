import logging
import numpy as np
import tensorflow as tf

from core.agent import AgentBase
from core.decorator import override
from core.mixin.agent import Memory


logger = logging.getLogger(__name__)


def infer_life_mask(discount, concat=True):
    life_mask = np.logical_or(
        discount, 1-np.any(discount, 1, keepdims=True)).astype(np.float32)
    # np.testing.assert_equal(life_mask, mask)
    if concat:
        life_mask = np.concatenate(life_mask)
    return life_mask

def collect(buffer, env_stats, env_step, reset, reward, 
            discount, next_obs, **kwargs):
    if env_stats.use_life_mask:
        kwargs['life_mask'] = infer_life_mask(discount)
    kwargs['reward'] = np.concatenate(reward)
    # discount is zero only when all agents are done
    discount[np.any(discount, 1)] = 1
    kwargs['discount'] = np.concatenate(discount)
    buffer.add(**kwargs)

def get_data_format(*, env, batch_size, sample_size=None,
        store_state=False, state_size=None, **kwargs):
    obs_dtype = tf.uint8 if len(env.obs_shape) == 3 else tf.float32
    action_dtype = tf.int32 if env.is_action_discrete else tf.float32
    data_format = dict(
        obs=((None, sample_size, *env.obs_shape), obs_dtype),
        global_state=((None, sample_size, *env.global_state_shape), env.global_state_dtype),
        action=((None, sample_size, *env.action_shape), action_dtype),
        value=((None, sample_size), tf.float32), 
        traj_ret=((None, sample_size), tf.float32),
        advantage=((None, sample_size), tf.float32),
        logpi=((None, sample_size), tf.float32),
        mask=((None, sample_size), tf.float32),
    )
    if env.use_action_mask:
        data_format['action_mask'] = (
            (None, sample_size, env.action_dim), tf.bool)
    if env.use_life_mask:
        data_format['life_mask'] = ((None, sample_size), tf.float32)
        
    if store_state:
        dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        data_format.update({
            k: ((batch_size, v), dtype)
                for k, v in state_size._asdict().items()
        })

    return data_format

def random_actor(env_output, env=None, **kwargs):
    obs = env_output.obs
    a = np.concatenate(env.random_action())
    terms = {
        'obs': np.concatenate(obs['obs']), 
        'global_state': np.concatenate(obs['global_state']),
    }
    return a, terms


class Agent(AgentBase, Memory):
    """ Initialization """
    @override(AgentBase)
    def _post_init(self, env_stats, dataset):
        super()._post_init(env_stats, dataset)

        state_keys = self.model.state_keys
        mid = len(state_keys) // 2
        self._actor_state_keys = state_keys[:mid]
        self._value_state_keys = state_keys[mid:]
        self._value_sample_keys = [
            'global_state', 'value', 
            'traj_ret', 'mask'
        ] + list(self._value_state_keys)
        if env_stats.use_life_mask:
            self._value_sample_keys.append('life_mask')


    # @override(PPOBase)
    # def _summary(self, data, terms):
    #     tf.summary.histogram('sum/value', data['value'], step=self._env_step)
    #     tf.summary.histogram('sum/logpi', data['logpi'], step=self._env_step)

    """ PPO methods """
    # @override(PPOBase)
    def record_last_env_output(self, env_output):
        self._env_output = self.model.reshape_env_output(env_output)
        obs = self.actor.process_obs_with_rms(self._env_output.obs, update_rms=False)
        self._mask = self._get_mask(self._env_output.reset)
        self._state = self._apply_mask_to_state(self._state, self._mask)

    def compute_value(self, global_state=None, state=None, mask=None, return_state=False):
        # be sure obs is normalized if obs normalization is required
        if global_state is None:
            global_state = self._env_output.obs['global_state']
        if state is None:
            state = self._state
        mid = len(self._state) // 2
        state = state[mid:]
        if mask is None:
            mask = self._mask
        value = self.model.compute_value(
            global_state=global_state, 
            state=state,
            mask=mask,
            return_state=return_state,
        )
        value = value.reshape(-1, self._n_agents)
        return value

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

        for _ in range(self.N_VALUE_EPOCHS):
            for _ in range(self.N_MBS):
                data = self.dataset.sample(self._value_sample_keys)

                data = {k: tf.convert_to_tensor(data[k]) 
                    for k in self._value_sample_keys}

                terms = self.trainer.learn_value(**data)
                terms = {f'train/{k}': v.numpy() for k, v in terms.items()}
                self.store(**terms)
        
        return n

    def _store_buffer_stats(self):
        pass
        # self.store(**self.dataset.compute_mean_max_std('reward'))
        # self.store(**self.dataset.compute_mean_max_std('obs'))
        # self.store(**self.dataset.compute_mean_max_std('global_state'))
        # self.store(**self.dataset.compute_mean_max_std('advantage'))
        # self.store(**self.dataset.compute_mean_max_std('value'))
        # self.store(**self.dataset.compute_mean_max_std('traj_ret'))
        # self.store(**self.dataset.compute_fraction('mask'))
        # self.store(**self.dataset.compute_fraction('discount'))
