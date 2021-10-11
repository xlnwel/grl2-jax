import os
import cloudpickle
import logging
import numpy as np
import tensorflow as tf

from core.log import do_logging
from utility.schedule import PiecewiseSchedule

logger = logging.getLogger(__name__)


""" Agent Mixins """
class StepCounter:
    def __init__(self, root_dir, model_name):
        self._env_step = 0
        self._train_step = 0
        self._counter_path = f'{root_dir}/{model_name}/step_counter.pkl'

    def get_env_step(self):
        return self._env_step

    def set_env_step(self, step):
        self._env_step = step

    def get_train_step(self):
        return self._train_step

    def set_train_step(self, step):
        self._train_step = step

    def save_step(self):
        with open(self._counter_path, 'wb') as f:
            cloudpickle.dump((self._env_step, self._train_step), f)

    def restore_step(self):
        if os.path.exists(self._counter_path):
            with open(self._counter_path, 'rb') as f:
                self._env_step, self._train_step = cloudpickle.load(f)


class ActionScheduler:
    def _setup_action_schedule(self, env):
        # eval action epsilon and temperature
        self._eval_act_eps = tf.convert_to_tensor(
            getattr(self, '_eval_act_eps', 0), tf.float32)
        self._eval_act_temp = tf.convert_to_tensor(
            getattr(self, '_eval_act_temp', .5), tf.float32)

        self._schedule_act_eps = getattr(self, '_schedule_act_eps', False)
        self._schedule_act_temp = getattr(self, '_schedule_act_temp', False)
        
        self._schedule_act_epsilon(env)
        self._schedule_act_temperature(env)

    def _schedule_act_epsilon(self, env):
        """ Schedules action epsilon """
        if self._schedule_act_eps:
            if isinstance(self._act_eps, (list, tuple)):
                do_logging(f'Schedule action epsilon: {self._act_eps}', logger=logger)
                self._act_eps = PiecewiseSchedule(self._act_eps)
            else:
                from utility.rl_utils import compute_act_eps
                self._act_eps = compute_act_eps(
                    self._act_eps_type, 
                    self._act_eps, 
                    getattr(self, '_id', None), 
                    getattr(self, '_n_workers', getattr(env, 'n_workers', 1)), 
                    env.n_envs)
                if env.action_shape != ():
                    self._act_eps = self._act_eps.reshape(-1, 1)
                self._schedule_act_eps = False  # not run-time scheduling
        print('Action epsilon:', np.reshape(self._act_eps, -1))
        if not isinstance(getattr(self, '_act_eps', None), PiecewiseSchedule):
            self._act_eps = tf.convert_to_tensor(self._act_eps, tf.float32)

    def _schedule_act_temperature(self, env):
        """ Schedules action temperature """
        if self._schedule_act_temp:
            from utility.rl_utils import compute_act_temp
            self._act_temp = compute_act_temp(
                self._min_temp,
                self._max_temp,
                getattr(self, '_n_exploit_envs', 0),
                getattr(self, '_id', None),
                getattr(self, '_n_workers', getattr(env, 'n_workers', 1)), 
                env.n_envs)
            self._act_temp = self._act_temp.reshape(-1, 1)
            self._schedule_act_temp = False         # not run-time scheduling    
        else:
            self._act_temp = getattr(self, '_act_temp', 1)
        print('Action temperature:', np.reshape(self._act_temp, -1))
        self._act_temp = tf.convert_to_tensor(self._act_temp, tf.float32)

    def _get_eps(self, evaluation):
        """ Gets action epsilon """
        if evaluation:
            eps = self._eval_act_eps
        else:
            if self._schedule_act_eps:
                eps = self._act_eps.value(self.env_step)
                self.store(act_eps=eps)
                eps = tf.convert_to_tensor(eps, tf.float32)
            else:
                eps = self._act_eps
        return eps
    
    def _get_temp(self, evaluation):
        """ Gets action temperature """
        return self._eval_act_temp if evaluation else self._act_temp


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
