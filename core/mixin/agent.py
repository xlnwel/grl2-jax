import logging
import numpy as np
import tensorflow as tf

from core.log import do_logging
from utility.schedule import PiecewiseSchedule

logger = logging.getLogger(__name__)


""" Agent Mixins """
class ActionScheduler:
    # NOTE: need some extra fix
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

