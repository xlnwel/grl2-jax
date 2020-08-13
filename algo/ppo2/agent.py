import numpy as np
import tensorflow as tf

from core.tf_config import build
from core.decorator import agent_config
from nn.rnn import LSTMState
from algo.ppo.base import PPOBase
from algo.ppo.loss import compute_ppo_loss, compute_value_loss


class Agent(PPOBase):
    @agent_config
    def __init__(self, *, dataset, env):
        super().__init__(dataset=dataset, env=env)

        # previous and current state of LSTM
        self.state = self.ac.get_initial_state(batch_size=env.n_envs)
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=((self._sample_size, *env.obs_shape), env.obs_dtype, 'obs'),
            action=((self._sample_size, *env.action_shape), env.action_dtype, 'action'),
            traj_ret=((self._sample_size,), tf.float32, 'traj_ret'),
            value=((self._sample_size,), tf.float32, 'value'),
            advantage=((self._sample_size,), tf.float32, 'advantage'),
            logpi=((self._sample_size,), tf.float32, 'logpi'),
            mask=((self._sample_size,), tf.float32, 'mask'),
        )
        if self._store_state:
            state_type = type(self.ac.state_size)
            TensorSpecs['state'] = state_type(*[((sz, ), self._dtype, name) 
                for name, sz in self.ac.state_size._asdict().items()])
        self.learn = build(self._learn, TensorSpecs, print_terminal_info=True)

    def reset_states(self, states=None):
        self.state = states

    def get_states(self):
        return self.state

    def __call__(self, obs, reset=None, deterministic=False, 
                update_curr_state=True, update_rms=False, **kwargs):
        if len(obs.shape) % 2 != 0:
            obs = np.expand_dims(obs, 0)    # add batch dimension
        assert len(obs.shape) in (2, 4), obs.shape  
        if self.state is None:
            self.state = self.ac.get_initial_state(batch_size=tf.shape(obs)[0])
        if reset is None:
            mask = tf.ones(tf.shape(obs)[0], dtype=tf.float32)
        else:
            mask = tf.cast(1. - reset, tf.float32)
        if update_rms:
            self.update_obs_rms(obs)
        obs = self.normalize_obs(obs)
        prev_state = self.state
        out, state = self.model.action(obs, self.state, mask, deterministic)
        if update_curr_state:
            self.state = state
        if deterministic:
            return out.numpy()
        else:
            action, terms = out
            terms['mask'] = mask
            if self._store_state:
                terms.update(prev_state._asdict())
            terms = tf.nest.map_structure(lambda x: x.numpy(), terms)
            terms['obs'] = obs  # return normalized obs
            return action.numpy(), terms
