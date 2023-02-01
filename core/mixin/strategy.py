import os
from typing import NamedTuple
import cloudpickle
import numpy as np

from core.typing import ModelPath
from jax_tools import jax_utils


class StepCounter:
    def __init__(self, model_path: ModelPath, name='step_counter'):
        self._env_step = 0
        self._train_step = 0
        self._env_step_interval = 0
        self._train_step_interval = 0
        self._path = None if model_path is None else '/'.join([*model_path, f'{name}.pkl'])

    def get_env_step(self):
        return self._env_step

    def set_env_step(self, step):
        self._env_step_interval = step - self._env_step
        self._env_step = step

    def add_env_step(self, steps):
        self._env_step_interval = steps
        self._env_step += steps

    def get_train_step(self):
        return self._train_step

    def set_train_step(self, step):
        self._train_step_interval = step - self._train_step
        self._train_step = step

    def add_train_step(self, steps):
        self._train_step_interval = steps
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

    def get_env_step_intervals(self):
        return self._env_step_interval

    def get_train_step_intervals(self):
        return self._train_step_interval

class Memory:
    def __init__(self, model):
        """ Setups attributes for RNNs """
        self.model = model
        self._state = None

    def add_memory_state_to_input(self, 
            inp: dict, reset: np.ndarray, state: NamedTuple=None):
        """ Adds memory state and mask to the input. """
        if state is None and self._state is None:
            self._state = self.model.get_initial_state(
                next(iter(inp.values())).shape[0])

        if state is None:
            state = self._state

        state = self.apply_reset_to_state(state, reset)
        inp.update({
            'state_reset': reset, 
            'state': state,
        })

        return inp

    def get_mask(self, reset):
        return np.float32(1. - reset)

    def apply_reset_to_state(self, state: NamedTuple, reset: np.ndarray):
        assert state is not None, state
        reset = reset.reshape(-1, 1)
        state = jax_utils.tree_map(lambda x: x*(1-reset), state)
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
