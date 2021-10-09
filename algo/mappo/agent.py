import logging
from typing import Dict
import numpy as np
import tensorflow as tf

from core.decorator import override
from core.mixin.agent import Memory
from utility.utils import concat_map
from algo.ppo.agent import PPOAgent

logger = logging.getLogger(__name__)


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


class MAPPOAgent(PPOAgent):
    """ Initialization """
    @override(PPOAgent)
    def _post_init(self, env_stats, dataset):
        super()._post_init(env_stats, dataset)
        self._memory = Memory(self.model)

        state_keys = self.model.state_keys
        mid = len(state_keys) // 2
        value_state_keys = state_keys[mid:]
        self._value_sample_keys = [
            'global_state', 'value', 'traj_ret', 'mask'
        ] + list(value_state_keys)
        if env_stats.use_life_mask:
            self._value_sample_keys.append('life_mask')
        self._n_agents = env_stats.n_agents

    # @override(PPOBase)
    # def _summary(self, data, terms):
    #     tf.summary.histogram('sum/value', data['value'], step=self._env_step)
    #     tf.summary.histogram('sum/logpi', data['logpi'], step=self._env_step)
    """ Training Methods """
    def _train_extra_vf(self):
        for _ in range(self.N_VALUE_EPOCHS):
            for _ in range(self.N_MBS):
                data = self.dataset.sample(self._value_sample_keys)

                data = {k: tf.convert_to_tensor(data[k]) 
                    for k in self._value_sample_keys}

                terms = self.trainer.learn_value(**data)
                terms = {f'train/{k}': v.numpy() for k, v in terms.items()}
                self.store(**terms)

    """ Calling Methods """
    def _prepare_input_to_actor(self, env_output):
        inp = env_output.obs
        inp['discount'] = env_output.discount
        inp = self._memory.add_memory_state_to_input(inp, env_output.reset)

        return inp

    def _record_output(self, out):
        state = out[-1]
        self._memory.reset_states(state)

    """ PPO Methods """
    def record_inputs_to_vf(self, env_output):
        value_input = concat_map({'global_state': env_output.obs['global_state']})
        value_input = self.actor.process_obs_with_rms(
            value_input, update_rms=False)
        reset = concat_map(env_output.reset)
        state = self._memory.get_states()
        mid = len(state) // 2
        state = self.model.value_state_type(*state[mid:])
        self._value_input = self._memory.add_memory_state_to_input(
            value_input, reset, state=state)

    def compute_value(self, value_inp: Dict[str, np.ndarray]=None):
        # be sure global_state is normalized if obs normalization is required
        if value_inp is None:
            value_inp = self._value_input
        value, _ = self.model.compute_value(**value_inp)
        value = value.numpy().reshape(-1, self._n_agents)
        return value

def create_agent(**kwargs):
    return MAPPOAgent(**kwargs)
