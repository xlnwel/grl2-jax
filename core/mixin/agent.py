import cloudpickle
import logging
import numpy as np
import tensorflow as tf

from core.log import *
from utility.schedule import PiecewiseSchedule

logger = logging.getLogger(__name__)


""" Agent Mixins """
class StepCounter:
    def _initialize_counter(self):
        self.env_step = 0
        self.train_step = 0
        self._counter_path = f'{self._root_dir}/{self._model_name}/counter.pkl'

    def save_step(self):
        with open(self._counter_path, 'wb') as f:
            cloudpickle.dump((self.env_step, self.train_step), f)

    def restore_step(self):
        if os.path.exists(self._counter_path):
            with open(self._counter_path, 'rb') as f:
                self.env_step, self.train_step = cloudpickle.load(f)


class TensorboardOps:
    """ Tensorboard Ops """
    def set_summary_step(self, step):
        """ Sets tensorboard step """
        set_summary_step(step)

    def scalar_summary(self, stats, prefix=None, step=None):
        """ Adds scalar summary to tensorboard """
        scalar_summary(self._writer, stats, prefix=prefix, step=step)

    def histogram_summary(self, stats, prefix=None, step=None):
        """ Adds histogram summary to tensorboard """
        histogram_summary(self._writer, stats, prefix=prefix, step=step)

    def graph_summary(self, sum_type, *args, step=None):
        """ Adds graph summary to tensorboard
        Args:
            sum_type str: either "video" or "image"
            args: Args passed to summary function defined in utility.graph,
                of which the first must be a str to specify the tag in Tensorboard
        """
        assert isinstance(args[0], str), f'args[0] is expected to be a name string, but got "{args[0]}"'
        args = list(args)
        args[0] = f'{self.name}/{args[0]}'
        graph_summary(self._writer, sum_type, args, step=step)

    def video_summary(self, video, step=None):
        video_summary(f'{self.name}/sim', video, step=step)

    def save_config(self, config):
        """ Save config.yaml """
        save_config(self._root_dir, self._model_name, config)


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
                logger.info(f'Schedule action epsilon: {self._act_eps}')
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
    def _setup_memory_state_record(self):
        """ Setups attributes for RNNs """
        self._state = None

    def _get_state_with_batch_size(self, batch_size):
        return self.model.get_initial_state(batch_size=batch_size)

    def _prepare_input_to_actor(self, env_output):
        inp = env_output.obs
        mask = self._get_mask(env_output.reset)
        inp = self._add_memory_state_to_input(inp, mask)

        return inp

    def _add_memory_state_to_input(self, 
            inp: dict, mask: np.ndarray, state=None, prev_reward=None, batch_size=None):
        """ Adds memory state to the input. Call this in self._process_input 
        when introducing sequential memory.
        """
        if state is None and self._state is None:
            batch_size = batch_size or mask.shape[0]
            self._state = self.model.get_initial_state(batch_size=batch_size)

        if state is None:
            state = self._state

        state = self._apply_mask_to_state(state, mask)
        inp.update({
            'state': state,
            'mask': mask,   # mask is applied in RNN
        })

        return inp

    def _record_output(self, out):
        _, _, self._state = out

    def _get_mask(self, reset):
        return np.float32(1. - reset)

    def _apply_mask_to_state(self, state, mask):
        if state is not None:
            mask_exp = np.expand_dims(mask, -1)
            if isinstance(state, (list, tuple)):
                state_type = type(state)
                state = state_type(*[v * mask_exp for v in state])
            else:
                state = state * mask_exp
        return state

    def reset_states(self, state=None):
        self._state = state

    def get_states(self):
        return self._state
