import numpy as np
import tensorflow as tf

from utility.display import pwc
from utility.schedule import PiecewiseSchedule
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config
from algo.ppo.loss import compute_ppo_loss, compute_value_loss


class Agent(BaseAgent):
    @agent_config
    def __init__(self, 
                name, 
                config, 
                models, 
                state_shape,
                state_dtype,
                action_dim,
                action_dtype,
                n_envs):
        # optimizer
        if getattr(self, 'schedule_lr', False):
            self.learning_rate = tf.Variable(self.learning_rate, trainable=False)
            self.schedule = PiecewiseSchedule(
                [(300, self.learning_rate), (1000, 5e-5)])
        if self.optimizer.lower() == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                epsilon=self.epsilon
            )
        elif self.optimizer.lower() == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=self.learning_rate,
                rho=.99,
                epsilon=self.epsilon
            )
        else:
            raise NotImplementedError()

        self.ckpt_models['optimizer'] = self.optimizer

        # initial, previous, and current state of LSTM
        self.initial_states = self.ac.get_initial_state()

        self.prev_states = self.initial_states   # for training
        self.curr_states = self.initial_states   # for environment interaction

        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = [
            (state_shape, state_dtype, 'state'),
            ([action_dim], action_dtype, 'action'),
            ([1], tf.float32, 'traj_ret'),
            ([1], tf.float32, 'value'),
            ([1], tf.float32, 'advantage'),
            ([1], tf.float32, 'old_logpi'),
            ([1], tf.float32, 'mask'),
            ((), tf.float32, 'n'),
        ]
        self.learn = build(
            self._learn, 
            TensorSpecs, 
            sequential=True, 
            batch_size=n_envs,
        )

    def reset_states(self):
        self.prev_states = self.curr_states = self.initial_states

    def step(self, state, update_curr_states=True):
        state = tf.convert_to_tensor(state, tf.float32)
        action, logpi, value, states = self.ac.step(state, self.curr_states)
        if update_curr_states:
            self.curr_states = states
        return action, logpi, value

    def det_action(self, state, update_curr_states=True):
        state = tf.convert_to_tensor(state, tf.float32)
        action, states = self.ac.det_action(state, self.curr_states)
        if update_curr_states:
            self.curr_states = states
        return action

    def learn_log(self, buffer, epoch):
        if not isinstance(self.learning_rate, float):
            self.learning_rate.assign(self.schedule.value(epoch))
        for i in range(self.n_updates):
            data = buffer.sample()
            data['n'] = n = np.sum(data['mask'])
            value = data['value']
            data = {k: tf.convert_to_tensor(v) for k, v in data.items()}
            with tf.name_scope('train'):
                terms  = self.learn(**data)
                
            n_total_trans = value.size
            n_valid_trans = n or n_total_trans

            terms['value'] = np.mean(value)
            terms['n_valid_trans'] = n_valid_trans
            terms['n_total_trans'] = n_total_trans
            terms['valid_trans_frac'] = n_valid_trans / n_total_trans
            
            approx_kl = terms['approx_kl']
            del terms['approx_kl']

            self.store(**terms)

            if getattr(self, 'max_kl', 0) > 0 and approx_kl > self.max_kl:
                pwc(f'Eearly stopping at update-{i+1} due to reaching max kl.',
                    f'Current kl={approx_kl:.3g}', color='blue')
                break
        self.store(approx_kl=approx_kl)
        if not isinstance(self.learning_rate, float):
            self.store(learning_rate=self.learning_rate.numpy())

        # update the states with the newest weights 
        states = self.ac.rnn_states(data['state'], *self.prev_states)
        self.prev_states = self.curr_states = states

    @tf.function
    def _learn(self, state, action, traj_ret, value, advantage, old_logpi, mask=None, n=None):
        with tf.GradientTape() as tape:
            old_value = value
            logpi, entropy, value, logstd = self.ac.train_step(state, action, self.prev_states)
            # policy loss
            ppo_loss, entropy, approx_kl, p_clip_frac = compute_ppo_loss(
                logpi, old_logpi, advantage, self.clip_range,
                entropy, mask=mask, n=n)
            # value loss
            value_loss, v_clip_frac = compute_value_loss(
                value, traj_ret, old_value, self.clip_range,
                mask=mask, n=n)

            with tf.name_scope('total_loss'):
                policy_loss = (ppo_loss 
                        - self.entropy_coef * entropy # TODO: adaptive entropy regularizer
                        + self.kl_coef * approx_kl)
                value_loss = self.value_coef * value_loss
                total_loss = policy_loss + value_loss

        with tf.name_scope('gradient'):
            grads = tape.gradient(total_loss, self.ac.trainable_variables)
            if hasattr(self, 'clip_norm'):
                grads, global_norm = tf.clip_by_global_norm(grads, self.clip_norm)
            self.optimizer.apply_gradients(zip(grads, self.ac.trainable_variables))

        terms = dict(
            entropy=entropy, 
            approx_kl=approx_kl, 
            p_clip_frac=p_clip_frac,
            v_clip_frac=v_clip_frac,
            ppo_loss=ppo_loss,
            value_loss=value_loss,
            global_norm=global_norm,
        )
        if logstd is not None:
            terms['std'] = tf.exp(logstd)
        return terms
