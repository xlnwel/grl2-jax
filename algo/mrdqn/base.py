import tensorflow as tf

from core.tf_config import build
from core.decorator import override
from core.base import Memory
from algo.dqn.base import DQNBase


def get_data_format(*, env, replay_config, agent_config,
        model, **kwargs):
    is_per = replay_config['replay_type'].endswith('per')
    store_state = agent_config['store_state']
    sample_size = agent_config['sample_size']
    obs_dtype = tf.uint8 if len(env.obs_shape) == 3 else tf.float32
    data_format = dict(
        obs=((None, sample_size+1, *env.obs_shape), obs_dtype),
        action=((None, sample_size+1, *env.action_shape), tf.int32),
        reward=((None, sample_size), tf.float32), 
        mu=((None, sample_size+1), tf.float32),
        discount=((None, sample_size), tf.float32),
        mask=((None, sample_size+1), tf.float32),
    )
    if is_per:
        data_format['idxes'] = ((None), tf.int32)
        if replay_config.get('use_is_ratio'):
            data_format['IS_ratio'] = ((None, ), tf.float32)
    if store_state:
        state_size = model.state_size
        from tensorflow.keras.mixed_precision import global_policy
        state_dtype = global_policy().compute_dtype
        data_format.update({
            k: ((None, v), state_dtype)
                for k, v in state_size._asdict().items()
        })

    return data_format

def collect(replay, env, step, reset, next_obs, **kwargs):
    replay.add(**kwargs)


class RDQNBase(Memory, DQNBase):
    """ Initialization """
    @override(DQNBase)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)
        self._burn_in = 'rnn' in self.model and self._burn_in
        self._setup_memory_state_record()

    @override(DQNBase)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to initialize variables
        TensorSpecs = dict(
            obs=((self._sample_size+1, *env.obs_shape), env.obs_dtype, 'obs'),
            action=((self._sample_size+1, env.action_dim), tf.float32, 'action'),
            reward=((self._sample_size,), tf.float32, 'reward'),
            mu=((self._sample_size+1,), tf.float32, 'mu'),
            discount=((self._sample_size,), tf.float32, 'discount'),
            mask=((self._sample_size+1,), tf.float32, 'mask')
        )
        if self._is_per and getattr(self, '_use_is_ratio', self._is_per):
            TensorSpecs['IS_ratio'] = ((), tf.float32, 'IS_ratio')
        if self._store_state:
            state_type = type(self.model.state_size)
            TensorSpecs['state'] = state_type(*[((sz, ), self._dtype, name) 
                for name, sz in self.model.state_size._asdict().items()])
        if self.model.additional_rnn_input:
            TensorSpecs['additional_rnn_input'] = [(
                ((self._sample_size, env.action_dim), self._dtype, 'prev_action'),
                ((self._sample_size, 1), self._dtype, 'prev_reward'),    # this reward should be unnormlaized
            )]
        self.learn = build(self._learn, TensorSpecs, batch_size=self._batch_size)

    """ Call """
    # @override(DQNBase)
    def _process_input(self, obs, evaluation, env_output):
        obs, kwargs = super()._process_input(obs, evaluation, env_output)
        obs, kwargs = self._add_memory_state_to_kwargs(obs, env_output, kwargs)
        return obs, kwargs

    # @override(DQNBase)
    def _process_output(self, obs, kwargs, out, evaluation):
        out = self._add_tensor_memory_state_to_terms(obs, kwargs, out, evaluation)
        out = super()._process_output(obs, kwargs, out, evaluation)
        out = self._add_non_tensor_memory_states_to_terms(out, kwargs, evaluation)
        return out

    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        priority = (self._per_eta*tf.math.reduce_max(priority, axis=1) 
                    + (1-self._per_eta)*tf.math.reduce_mean(priority, axis=1))
        priority += self._per_epsilon
        priority **= self._per_alpha
        return priority

    def _compute_embed(self, obs, mask, state, add_inp, online=True):
        encoder = self.encoder if online else self.target_encoder
        x = encoder(obs)
        if 'rnn' in self.model:
            rnn = self.rnn if online else self.target_rnn
            x, state = rnn(x, state, mask, additional_input=add_inp)
        return x, state