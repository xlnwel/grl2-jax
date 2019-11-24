import numpy as np
import tensorflow as tf

from utility.display import pwc
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config


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
        self.compute_gradients = build(
            self._compute_gradients, 
            TensorSpecs, 
            sequential=True, 
            batch_size=n_envs
        )

    def train_epoch(self, buffer, early_terminate, epoch):
        for i in range(self.n_updates):
            self.ac.reset_states()
            for j in range(self.n_minibatches):
                data = buffer.get_batch()
                data['n'] = n = np.sum(data['mask'])
                value = data['value']
                data = {k: tf.convert_to_tensor(v) for k, v in data.items()}
                with tf.name_scope('train'):
                    loss_info  = self.compute_gradients(**data)
                    entropy, approx_kl, p_clip_frac, v_clip_frac, ppo_loss, value_loss, grads = loss_info
                    if hasattr(self, 'clip_norm'):
                        grads, global_norm = tf.clip_by_global_norm(grads, self.clip_norm)
                    self.optimizer.apply_gradients(zip(grads, self.ac.trainable_variables))

                n_total_trans = value.shape
                n_valid_trans = n or n_total_trans
                self.store(
                    ppo_loss=ppo_loss.numpy(), 
                    value_loss=value_loss.numpy(),
                    entropy=entropy.numpy(), 
                    p_clip_frac=p_clip_frac.numpy(),
                    v_clip_frac=v_clip_frac.numpy(),
                    value=np.mean(value),
                    global_norm=global_norm.numpy(),
                    n_valid_trans=n_valid_trans,
                    n_total_trans=n_total_trans,
                    valid_trans_frac = n_valid_trans / n_total_trans
                )
            
            if self.max_kl and early_terminate and approx_kl > self.max_kl:
                pwc(f'Eearly stopping at epoch-{epoch} update-{i+1} due to reaching max kl.',
                    f'Current kl={approx_kl:.3g}', color='blue')
                break
        self.store(approx_kl=approx_kl)

    @tf.function
    def _compute_gradients(self, state, action, traj_ret, value, advantage, old_logpi, mask=None, n=None):
        with tf.GradientTape() as tape:
            logpi, entropy, v = self.ac.train_step(state, action)
            loss_info = self._loss(
                logpi, old_logpi, advantage, v, 
                traj_ret, value, self.clip_range,
                entropy, mask=mask, n=n)
            ppo_loss, entropy, approx_kl, p_clip_frac, value_loss, v_clip_frac, total_loss = loss_info

        with tf.name_scope('gradient'):
            grads = tape.gradient(total_loss, self.ac.trainable_variables)

        return entropy, approx_kl, p_clip_frac, v_clip_frac, ppo_loss, value_loss, grads 

    def _loss(self, logpi, old_logpi, advantages, value, traj_ret, old_value, clip_range, entropy, mask=None, n=None):
        assert (mask is None) == (n is None), f'Both/Neither mask and/nor n should be None, but get \nmask:{mask}\nn:{n}'

        def reduce_mean(x, name, n):
            with tf.name_scope(name):        
                return tf.reduce_mean(x) if n is None else tf.reduce_sum(x) / n

        m = 1. if mask is None else mask
        with tf.name_scope('ppo_loss'):
            ratio = tf.exp(logpi - old_logpi, name='ratio')
            loss1 = -advantages * ratio
            loss2 = -advantages * tf.clip_by_value(ratio, 1. - clip_range, 1. + clip_range)
            
            ppo_loss = reduce_mean(tf.maximum(loss1, loss2) * m, 'ppo_loss', n)
            entropy = tf.reduce_mean(entropy, name='entropy_loss')
            # debug stats: KL between old and current policy and fraction of data being clipped
            approx_kl = .5 * reduce_mean((old_logpi - logpi)**2 * m, 'approx_kl', n)
            p_clip_frac = reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.), clip_range), tf.float32) * m, 
                                    'clip_frac', n)
            policy_loss = (ppo_loss 
                        - self.entropy_coef * entropy # TODO: adaptive entropy regularizer
                        + self.kl_coef * approx_kl)

        with tf.name_scope('value_loss'):
            value_clipped = old_value + tf.clip_by_value(value - old_value, -clip_range, clip_range)
            loss1 = (value - traj_ret)**2
            loss2 = (value_clipped - traj_ret)**2
            
            value_loss = self.value_coef * reduce_mean(tf.maximum(loss1, loss2) * m, 'value_loss', n)
            v_clip_frac = reduce_mean(
                tf.cast(tf.greater(tf.abs(value-old_value), clip_range), tf.float32) * m,
                'clip_frac', n)
        
        with tf.name_scope('total_loss'):
            total_loss = policy_loss + value_loss

        return ppo_loss, entropy, approx_kl, p_clip_frac, value_loss, v_clip_frac, total_loss
