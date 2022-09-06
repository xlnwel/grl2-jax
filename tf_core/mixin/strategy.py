import os
from typing import NamedTuple
import cloudpickle
import numpy as np
import tensorflow as tf

from core.typing import ModelPath


class StepCounter:
    def __init__(self, model_path: ModelPath, name='step_counter'):
        self._env_step = 0
        self._train_step = 0
        self._path = None if model_path is None else '/'.join([*model_path, f'{name}.pkl'])

    def get_env_step(self):
        return self._env_step

    def set_env_step(self, step):
        self._env_step = step

    def add_env_step(self, steps):
        self._env_step += steps

    def get_train_step(self):
        return self._train_step

    def set_train_step(self, step):
        self._train_step = step

    def add_train_step(self, steps):
        self._train_step += steps

    def get_steps(self):
        return self._train_step, self._env_step

    def set_steps(self, steps):
        self._train_step, self._env_step = steps
    
    def save_step(self):
        if self._path:
            with open(self._path, 'wb') as f:
                cloudpickle.dump((self._env_step, self._train_step), f)

    def restore_step(self):
        if self._path is not None and os.path.exists(self._path):
            with open(self._path, 'rb') as f:
                self._env_step, self._train_step = cloudpickle.load(f)


class Memory:
    def __init__(self, model):
        """ Setups attributes for RNNs """
        self.model = model
        self._state = None

    def add_memory_state_to_input(self, 
            inp: dict, reset: np.ndarray, state: NamedTuple=None):
        """ Adds memory state and mask to the input. """
        if state is None and self._state is None:
            self._state = self.model.get_initial_state(inp)

        if state is None:
            state = self._state

        mask = self.get_mask(reset)
        state = self.apply_mask_to_state(state, mask)
        inp.update({
            'state': state,
            'mask': mask,   # mask is applied in RNN
        })

        return inp

    def get_mask(self, reset):
        return np.float32(1. - reset)

    def apply_mask_to_state(self, state: NamedTuple, mask: np.ndarray):
        assert state is not None, state
        mask_reshaped = mask.reshape(-1, 1)
        state = tf.nest.map_structure(lambda x: x*mask_reshaped \
            if x is not None else x, state)
        return state 

    def reset_states(self):
        self._state = None

    def set_states(self, state: NamedTuple=None):
        self._state = state

    def get_states(self):
        return self._state

    def get_states_for_inputs(self, **kwargs):
        if self._state is None:
            self._state = self.model.get_initial_state(**kwargs)
        return self._state
