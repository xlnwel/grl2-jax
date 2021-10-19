import os
import cloudpickle
import numpy as np

from utility.timer import Timer
from utility.utils import config_attr


class StepCounter:
    def __init__(self, root_dir, model_name, name='step_counter'):
        self._env_step = 0
        self._train_step = 0
        self._counter_path = f'{root_dir}/{model_name}/{name}.pkl'

    def get_env_step(self):
        return self._env_step

    def set_env_step(self, step):
        self._env_step = step

    def get_train_step(self):
        return self._train_step

    def set_train_step(self, step):
        self._train_step = step

    def get_steps(self):
        return self._train_step, self._env_step

    def set_steps(self, steps):
        self._train_step, self._env_step = steps
    
    def save_step(self):
        with open(self._counter_path, 'wb') as f:
            cloudpickle.dump((self._env_step, self._train_step), f)

    def restore_step(self):
        if os.path.exists(self._counter_path):
            with open(self._counter_path, 'rb') as f:
                self._env_step, self._train_step = cloudpickle.load(f)


class TrainingLoopBase:
    def __init__(self, 
                 config, 
                 dataset, 
                 trainer, 
                 **kwargs):
        config_attr(self, config)
        self.dataset = dataset
        self.trainer = trainer

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._sample_timer = Timer('sample')
        self._train_timer = Timer('train')
        self._post_init()

    def _post_init(self):
        pass

    def train(self):
        train_step, stats = self._train()
        self._after_train()

        return train_step, stats

    def _train(self):
        raise NotImplementedError

    def _after_train(self):
        pass


class Memory:
    def __init__(self, model):
        """ Setups attributes for RNNs """
        self.model = model
        self._state = None

    def add_memory_state_to_input(self, 
            inp: dict, reset: np.ndarray, state: tuple=None, batch_size: int=None):
        """ Adds memory state and mask to the input. """
        if state is None and self._state is None:
            batch_size = batch_size or reset.size
            self._state = self.model.get_initial_state(batch_size=batch_size)

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

    def apply_mask_to_state(self, state: tuple, mask: np.ndarray):
        if state is not None:
            mask_reshaped = mask.reshape(state[0].shape[0], 1)
            if isinstance(state, (list, tuple)):
                state_type = type(state)
                state = state_type(*[v * mask_reshaped for v in state])
            else:
                state = state * mask_reshaped
        return state

    def reset_states(self, state: tuple=None):
        self._state = state

    def get_states(self):
        return self._state
