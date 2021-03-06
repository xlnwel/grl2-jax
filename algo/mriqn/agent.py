import numpy as np
import tensorflow as tf

from utility.tf_utils import softmax, log_softmax
from utility.rl_utils import *
from utility.rl_loss import retrace
from core.tf_config import build
from core.decorator import override
from core.base import Memory
from algo.dqn.base import DQNBase


def get_data_format(*, env, replay_config, agent_config,
        model, **kwargs):
    is_per = replay_config['replay_type'].endswith('per')
    store_state = agent_config['store_state']
    sample_size = agent_config['sample_size']
    obs_dtype = tf.uint8 if len(env.obs_shape) == 3 else tf.float32
    data_format = dict(
        obs=((None, sample_size+1, *env.obs_shape), obs_dtype),
        action=((None, sample_size+1, *env.action_shape), tf.int32),
        reward=((None, sample_size), tf.float32), 
        prob=((None, sample_size+1), tf.float32),
        discount=((None, sample_size), tf.float32),
        mask=((None, sample_size+1), tf.float32),
    )
    if is_per:
        data_format['idxes'] = ((None), tf.int32)
        if replay_config.get('use_is_ratio'):
            data_format['IS_ratio'] = ((None, ), tf.float32)
    if store_state:
        state_size = model.state_size
        from tensorflow.keras.mixed_precision import global_policy
        state_dtype = global_policy().compute_dtype
        data_format.update({
            k: ((None, v), state_dtype)
                for k, v in state_size._asdict().items()
        })

    return data_format

def collect(replay, env, step, reset, next_obs, **kwargs):
    replay.add(**kwargs)


class Agent(Memory, DQNBase):
    """ Initialization """
    @override(DQNBase)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)
        self._burn_in = 'rnn' in self.model and self._burn_in
        self._setup_memory_state_record()

    @override(DQNBase)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to initialize variables
        TensorSpecs = dict(
            obs=((self._sample_size+1, *env.obs_shape), env.obs_dtype, 'obs'),
            action=((self._sample_size+1, env.action_dim), tf.float32, 'action'),
            reward=((self._sample_size,), tf.float32, 'reward'),
            prob=((self._sample_size+1,), tf.float32, 'prob'),
            discount=((self._sample_size,), tf.float32, 'discount'),
            mask=((self._sample_size+1,), tf.float32, 'mask')
        )
        if self._is_per and getattr(self, '_use_is_ratio', self._is_per):
            TensorSpecs['IS_ratio'] = ((), tf.float32, 'IS_ratio')
        if self._store_state:
            state_type = type(self.model.state_size)
            TensorSpecs['state'] = state_type(*[((sz, ), self._dtype, name) 
                for name, sz in self.model.state_size._asdict().items()])
        if self.model.additional_rnn_input:
            TensorSpecs['additional_rnn_input'] = [(
                ((self._sample_size, env.action_dim), self._dtype, 'prev_action'),
                ((self._sample_size, 1), self._dtype, 'prev_reward'),    # this reward should be unnormlaized
            )]
        self.learn = build(self._learn, TensorSpecs, batch_size=self._batch_size)

    """ Call """
    # @override(DQNBase)
    def _process_input(self, obs, evaluation, env_output):
        obs, kwargs = super()._process_input(obs, evaluation, env_output)
        obs, kwargs = self._add_memory_state_to_kwargs(obs, env_output, kwargs)
        return obs, kwargs

    # @override(DQNBase)
    def _process_output(self, obs, kwargs, out, evaluation):
        out = self._add_tensor_memory_state_to_terms(obs, kwargs, out, evaluation)
        out = super()._process_output(obs, kwargs, out, evaluation)
        out = self._add_non_tensor_memory_states_to_terms(out, kwargs, evaluation)
        return out

    """ MRDQN methods """
    @tf.function
    def _learn(self, obs, action, reward, discount, prob, mask, 
                IS_ratio=1, state=None, additional_rnn_input=[]):
        mask = tf.expand_dims(mask, -1)
        if additional_rnn_input != []:
            prev_action, prev_reward = additional_rnn_input
            prev_action = tf.concat([prev_action, action[:, :-1]], axis=1)
            prev_reward = tf.concat([prev_reward, reward[:, :-1]], axis=1)
            add_inp = [prev_action, prev_reward]
        else:
            add_inp = additional_rnn_input
            
        target, terms = self._compute_target(
            obs, action, reward, discount, 
            prob, mask, state, add_inp)
        if self._burn_in:
            bis = self._burn_in_size
            ss = self._sample_size - bis
            bi_obs, obs = tf.split(obs, [bis, ss], 1)
            bi_mask, mask = tf.split(mask, [bis, ss+1], 1)
            if add_inp:
                bi_add_inp, add_inp = zip(*[tf.split(v, [bis, ss+1]) for v in add_inp])
            else:
                bi_add_inp = []
            _, state = self._compute_embed(bi_obs, bi_mask, state, bi_add_inp)

        with tf.GradientTape() as tape:
            x, state = self._compute_embed(obs, mask, state, add_inp)
            
            curr_x = x[:, :-1]
            curr_action = action[:, :-1]

            
            q = self.q(curr_x, curr_action)
            error = target - q
            loss = tf.reduce_mean(.5 * error**2, axis=-1)
            loss = tf.reduce_mean(IS_ratio * loss)
        tf.debugging.assert_shapes([
            [q, (None, self._sample_size)],
            [target, (None, self._sample_size)],
            [error, (None, self._sample_size)],
            [IS_ratio, (None,)],
            [loss, ()]
        ])
        if self._is_per:
            priority = self._compute_priority(tf.abs(error))
            terms['priority'] = priority
        
        terms['norm'] = self._optimizer(tape, loss)
        
        terms.update(dict(
            q=q,
            prob_min=tf.reduce_min(prob),
            prob=prob,
            prob_std=tf.math.reduce_std(prob),
            target=target,
            loss=loss,
        ))

        return terms

    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        priority = (self._per_eta*tf.math.reduce_max(priority, axis=1) 
                    + (1-self._per_eta)*tf.math.reduce_mean(priority, axis=1))
        priority += self._per_epsilon
        priority **= self._per_alpha
        return priority

    def _compute_embed(self, obs, mask, state, add_inp, online=True):
        encoder = self.encoder if online else self.target_encoder
        x = encoder(obs)
        if 'rnn' in self.model:
            rnn = self.rnn if online else self.target_rnn
            x, state = rnn(x, state, mask, additional_input=add_inp)
        return x, state
    
    def _compute_target(self, obs, action, reward, discount, 
                        prob, mask, state, add_inp):
        terms = {}
        x, _ = self._compute_embed(obs, mask, state, add_inp, online=False)
        if self._burn_in:
            bis = self._burn_in_size
            ss = self._sample_size - bis
            _, reward = tf.split(reward, [bis, ss], 1)
            _, discount = tf.split(discount, [bis, ss], 1)
            _, next_mu_a = tf.split(prob, [bis+1, ss], 1)
            _, next_x = tf.split(x, [bis+1, ss], 1)
            _, next_action = tf.split(action, [bis+1, ss], 1)
        else:
            _, next_mu_a = tf.split(prob, [1, self._sample_size], 1)
            _, next_x = tf.split(x, [1, self._sample_size], 1)
            _, next_action = tf.split(action, [1, self._sample_size], 1)

        next_qs = self.target_q(next_x)
        if self._probabilistic_regularization is None:
            if self._double:
                online_x, _ = self._compute_embed(obs, mask, state, add_inp)
                next_online_x = tf.split(online_x, [bis+1, ss-1], 1)
                next_online_qs = self.q(next_online_x)
                next_pi = self.compute_greedy_action(next_online_qs, one_hot=True)
            else:    
                next_pi = self.target_q.compute_greedy_action(next_qs, one_hot=True)
        elif self._probabilistic_regularization == 'prob':
            next_pi = softmax(next_qs, self._tau)
        elif self._probabilistic_regularization == 'entropy':
            next_pi = softmax(next_qs, self._tau)
            next_logpi = log_softmax(next_qs, self._tau)
            neg_entropy = tf.reduce_sum(next_pi * next_logpi, axis=-1)
            terms['next_entropy'] = - neg_entropy / self._tau
        else:
            raise ValueError(self._probabilistic_regularization)

        discount = discount * self._gamma
        target = retrace(
            reward, next_qs, next_action, 
            next_pi, next_mu_a, discount,
            lambda_=self._lambda, 
            axis=1, tbo=self._tbo)

        return target, terms
