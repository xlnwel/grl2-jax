import logging
import numpy as np
import tensorflow as tf

from core.decorator import override
from core.mixin.agent import Memory
from utility.utils import concat_map
from algo.ppo.agent import Agent as PPOAgent

logger = logging.getLogger(__name__)


def infer_life_mask(discount, concat=True):
    life_mask = np.logical_or(
        discount, 1-np.any(discount, 1, keepdims=True)).astype(np.float32)
    # np.testing.assert_equal(life_mask, mask)
    if concat:
        life_mask = np.concatenate(life_mask)
    return life_mask

def collect(buffer, env, env_step, reset, reward, 
            discount, next_obs, **kwargs):
    if env.use_life_mask:
        kwargs['life_mask'] = infer_life_mask(discount)
    kwargs['reward'] = np.concatenate(reward)
    # discount is zero only when all agents are done
    discount[np.any(discount, 1)] = 1
    kwargs['discount'] = np.concatenate(discount)
    buffer.add(**kwargs)

def get_data_format(*, env_stats, batch_size, sample_size=None,
        store_state=False, state_size=None, **kwargs):
    obs_dtype = tf.uint8 if len(env_stats.obs_shape) == 3 else tf.float32
    action_dtype = tf.int32 if env_stats.is_action_discrete else tf.float32
    data_format = dict(
        obs=((None, sample_size, *env_stats.obs_shape), obs_dtype),
        global_state=((None, sample_size, *env_stats.global_state_shape), env_stats.global_state_dtype),
        action=((None, sample_size, *env_stats.action_shape), action_dtype),
        value=((None, sample_size), tf.float32), 
        traj_ret=((None, sample_size), tf.float32),
        advantage=((None, sample_size), tf.float32),
        logpi=((None, sample_size), tf.float32),
        mask=((None, sample_size), tf.float32),
    )
    if env_stats.use_action_mask:
        data_format['action_mask'] = (
            (None, sample_size, env_stats.action_dim), tf.bool)
    if env_stats.use_life_mask:
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


class Agent(Memory, PPOAgent):
    """ Initialization """
    @override(PPOAgent)
    def _post_init(self, env_stats, dataset):
        super()._post_init(env_stats, dataset)
        self._setup_memory_state_record()

        state_keys = self.model.state_keys
        mid = len(state_keys) // 2
        value_state_keys = state_keys[mid:]
        self._value_sample_keys = [
            'global_state', 'value', 
            'traj_ret', 'mask'
        ] + list(value_state_keys)
        if env_stats.use_life_mask:
            self._value_sample_keys.append('life_mask')
        self._n_agents = env_stats.n_agents

    # @override(PPOBase)
    # def _summary(self, data, terms):
    #     tf.summary.histogram('sum/value', data['value'], step=self._env_step)
    #     tf.summary.histogram('sum/logpi', data['logpi'], step=self._env_step)

    # def _prepare_input_to_actor(self, env_output):
    #     inp = concat_map(env_output.obs)
    #     mask = self._get_mask(concat_map(env_output.reset))
    #     inp = self._add_memory_state_to_input(inp, mask)

    #     return inp

    """ PPO methods """
    # @override(PPOBase)
    def record_last_env_output(self, env_output):
        global_state = concat_map(env_output.obs['global_state'])
        self._global_state = self.actor.process_obs_with_rms(
            ('global_state', global_state), update_rms=False)
        reset = concat_map(env_output.reset)
        self._mask = self._get_mask(reset)
        self._state = self._apply_mask_to_state(self._state, self._mask)

    def compute_value(self, global_state=None, state=None, mask=None):
        # be sure global_state is normalized if obs normalization is required
        if global_state is None:
            global_state = self._global_state
        if state is None:
            state = self._state
        if mask is None:
            mask = self._mask
        mid = len(self._state) // 2
        state = state[mid:]
        value, _ = self.model.compute_value(
            global_state=global_state, 
            state=state,
            mask=mask,
        )
        value = value.numpy().reshape(-1, self._n_agents)
        return value
