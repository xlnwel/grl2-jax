import numpy as np
import tensorflow as tf

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
        data_format['IS_ratio'] = ((None), tf.float32)
        data_format['idxes'] = ((None), tf.int32)
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
        DQNBase._add_attributes(self, env, dataset)
        Memory._add_attributes(self)

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
        if self._is_per:
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
        obs, kwargs = DQNBase._process_input(self, obs, evaluation, env_output)
        obs, kwargs = Memory._process_input(self, obs, env_output, kwargs)
        return obs, kwargs

    # @override(DQNBase)
    def _process_output(self, obs, kwargs, out, evaluation):
        out = Memory._process_output(self, obs, kwargs, out, evaluation)
        out = DQNBase._process_output(self, obs, kwargs, out, evaluation)
        if not evaluation:
            out[1]['mask'] = kwargs['mask']
        return out

    """ MRDQN methods """
    @tf.function
    def _learn(self, obs, action, reward, discount, prob, mask, 
                IS_ratio=1, state=None, additional_rnn_input=[]):
        terms = {}
        if additional_rnn_input != []:
            prev_action, prev_reward = additional_rnn_input
            prev_action = tf.concat([prev_action, action[:, :-1]], axis=1)
            prev_reward = tf.concat([prev_reward, reward[:, :-1]], axis=1)
            add_inp = [prev_action, prev_reward]
        else:
            add_inp = additional_rnn_input
        mask = tf.expand_dims(mask, -1)
        with tf.GradientTape() as tape:
            x = self.encoder(obs)
            t_x = self.target_encoder(obs)
            ss = self._sample_size
            if 'rnn' in self.model:
                if self._burn_in:
                    bis = self._burn_in_size
                    ss = self._sample_size - bis
                    bi_x, x = tf.split(x, [bis, ss+1], 1)
                    tbi_x, t_x = tf.split(t_x, [bis, ss+1], 1)
                    if add_inp != []:
                        bi_add_inp, add_inp = zip(
                            *[tf.split(v, [bis, ss+1]) for v in add_inp])
                    else:
                        bi_add_inp = []
                    bi_mask, mask = tf.split(mask, [bis, ss+1], 1)
                    bi_discount, discount = tf.split(discount, [bis, ss], 1)
                    _, prob = tf.split(prob, [bis, ss], 1)
                    _, o_state = self.rnn(bi_x, state, bi_mask,
                        additional_input=bi_add_inp)
                    _, t_state = self.target_rnn(tbi_x, state, bi_mask,
                        additional_input=bi_add_inp)
                    o_state = tf.nest.map_structure(tf.stop_gradient, o_state)
                else:
                    o_state = t_state = state

                x, _ = self.rnn(x, o_state, mask,
                    additional_input=add_inp)
                t_x, _ = self.target_rnn(t_x, t_state, mask,
                    additional_input=add_inp)
            
            curr_x = x[:, :-1]
            next_x = x[:, 1:]
            t_next_x = t_x[:, 1:]
            curr_action = action[:, :-1]
            next_action = action[:, 1:]
            discount = discount * self._gamma
            
            q = self.q(curr_x, curr_action)
            new_next_action = self.q.action(next_x)
            next_pi = tf.one_hot(new_next_action, self._action_dim, dtype=tf.float32)
            t_next_qs = self.target_q(t_next_x)
            next_mu_a = prob[:, 1:]
            target = retrace(
                reward, t_next_qs, next_action, 
                next_pi, next_mu_a, discount,
                lambda_=self._lambda, 
                axis=1, tbo=self._tbo)
            target = tf.stop_gradient(target)
            error = target - q
            loss = tf.reduce_mean(.5 * error**2, axis=-1)
            loss = tf.reduce_mean(IS_ratio * loss)
        tf.debugging.assert_shapes([
            [q, (None, ss)],
            [next_pi, (None, ss, self._action_dim)],
            [target, (None, ss)],
            [error, (None, ss)],
            [IS_ratio, (None,)],
            [loss, ()]
        ])
        if self._is_per:
            # we intend to use error as priority instead of TD error used in the paper
            priority = self._compute_priority(tf.abs(error))
            terms['priority'] = priority
        
        terms['norm'] = self._optimizer(tape, loss)
        
        terms.update(dict(
            q=q,
            prob=prob,
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
