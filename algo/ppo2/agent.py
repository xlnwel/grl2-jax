import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

from utility.display import pwc
from utility.schedule import TFPiecewiseSchedule
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config
from core.optimizer import Optimizer
from algo.ppo.loss import compute_ppo_loss, compute_value_loss


class Agent(BaseAgent):
    @agent_config
    def __init__(self, env):
        self._dtype = prec.global_policy().compute_dtype

        # optimizer
        if getattr(self, 'schedule_lr', False):
            self._learning_rate = TFPiecewiseSchedule(
                [(300, self._learning_rate), (1000, 5e-5)])

        self._optimizer = Optimizer(
            self._optimizer, self.ac, self._learning_rate, 
            clip_norm=self._clip_norm, epsilon=self._epsilon)
        self._ckpt_models['optimizer'] = self._optimizer

        # previous and current state of LSTM
        self.initial_state = self.ac.get_initial_state(batch_size=env.n_envs)
        self.prev_state = self.initial_state   # for training
        self.curr_state = self.initial_state   # for environment interaction

        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=(env.obs_shape, self._dtype, 'obs'),
            action=(env.action_shape, self._dtype, 'action'),
            traj_ret=((), self._dtype, 'traj_ret'),
            value=((), self._dtype, 'value'),
            advantage=((), self._dtype, 'advantage'),
            old_logpi=((), self._dtype, 'old_logpi'),
            mask=((), self._dtype, 'mask'),
            n=(None, self._dtype, 'n'),
            h=((), self._dtype, 'h'), 
            c=((), self._dtype, 'c')
        )
        self.learn = build(
            self._learn, 
            TensorSpecs, 
            sequential=True, 
            batch_size=env.n_envs,
        )

    def reset_states(self, states=None, reset=None):
        if states is not None:
            self.curr_state = states
        if reset is not None:
            mask = tf.cast(1. - reset, self._dtype)[:, None]
            self.prev_state = tf.nest.map_structure(lambda x: x * mask, self.prev_state)
            self.curr_state = tf.nest.map_structure(lambda x: x * mask, self.curr_state)
        
    def __call__(self, obs, deterministic=False, update_curr_state=True):
        obs = tf.convert_to_tensor(obs, self._dtype)
        if deterministic:
            action, state = self.ac.det_action(obs, self.curr_state)
            if update_curr_state:
                self.curr_state = state
            return action
        else:
            action, logpi, value, state = self.ac.step(obs, self.curr_state)
            if update_curr_state:
                self.curr_state = state
            return action, logpi, value

    def learn_log(self, buffer, step):
        for i in range(self.N_UPDATES):
            data = buffer.sample()
            data['n'] = n = np.sum(data['mask'])
            value = data['value']
            data = {k: tf.convert_to_tensor(v, self._dtype) for k, v in data.items()}
            with tf.name_scope('train'):
                state, terms = self.learn(**data, h=self.prev_state[0], c=self.prev_state[1])

            terms = {k: v.numpy() for k, v in terms.items()}
            n_total_trans = value.size
            n_valid_trans = n or n_total_trans

            terms['value'] = np.mean(value)
            terms['n_valid_trans'] = n_valid_trans
            terms['n_total_trans'] = n_total_trans
            terms['valid_trans_frac'] = n_valid_trans / n_total_trans
            
            approx_kl = terms['approx_kl']
            del terms['approx_kl']

            self.store(**terms)

            if getattr(self, '_max_kl', 0) > 0 and approx_kl > self._max_kl:
                pwc(f'Eearly stopping after {i+1} update(s) due to reaching max kl.',
                    f'Current kl={approx_kl:.3g}', color='blue')
                break
        self.store(approx_kl=approx_kl)
        if not isinstance(self._learning_rate, float):
            step = tf.cast(self.global_steps, self._dtype)
            self.store(learning_rate=self._learning_rate(step))

        # update the state with the newest weights 
        self.prev_state = self.curr_state = state

    @tf.function
    def _learn(self, obs, action, traj_ret, value, advantage, old_logpi, mask, n, h, c):
        old_value = value
        with tf.GradientTape() as tape:
            prev_state = [h, c]
            action_dist, value, state = self.ac.train_step(obs, prev_state)
            logpi = action_dist.log_prob(action)
            entropy = action_dist.entropy()
            # policy loss
            ppo_loss, entropy, approx_kl, p_clip_frac = compute_ppo_loss(
                logpi, old_logpi, advantage, self._clip_range,
                entropy, mask=mask, n=n)
            # value loss
            value_loss, v_clip_frac = compute_value_loss(
                value, traj_ret, old_value, self._clip_range,
                mask=mask, n=n)

            with tf.name_scope('total_loss'):
                policy_loss = (ppo_loss - self._entropy_coef * entropy)
                value_loss = self._value_coef * value_loss
                total_loss = policy_loss + value_loss

        terms = dict(
            entropy=entropy, 
            approx_kl=approx_kl, 
            p_clip_frac=p_clip_frac,
            v_clip_frac=v_clip_frac,
            ppo_loss=ppo_loss,
            value_loss=value_loss
        )
        terms['grads_norm'] = self._optimizer(tape, total_loss)

        return state, terms
