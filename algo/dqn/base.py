import numpy as np
import tensorflow as tf

from utility.utils import Every
from utility.schedule import PiecewiseSchedule
from utility.timer import TBTimer
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import override, step_track


def get_data_format(env, is_per=False, n_steps=1, dtype=tf.float32):
    obs_dtype = env.obs_dtype if len(env.obs_shape) == 3 else dtype
    action_dtype = tf.int32 if env.is_action_discrete else dtype
    data_format = dict(
        obs=((None, *env.obs_shape), obs_dtype),
        action=((None, *env.action_shape), action_dtype),
        reward=((None, ), dtype), 
        next_obs=((None, *env.obs_shape), obs_dtype),
        discount=((None, ), dtype),
    )
    if is_per:
        data_format['IS_ratio'] = ((None, ), dtype)
        data_format['idxes'] = ((None, ), tf.int32)
    if n_steps > 1:
        data_format['steps'] = ((None, ), dtype)

    return data_format


class DQNBase(BaseAgent):
    """ Initialization """
    @override(BaseAgent)
    def _add_attributes(self, env, dataset):
        self._is_per = dataset.name().endswith('per')
        self._return_stats = getattr(self, '_return_stats', False)
        self._schedule_act_eps = env.n_envs == 1 and self._schedule_act_eps

        if self._schedule_act_eps:
            if isinstance(self._act_eps, (float, int)):
                self._act_eps = [(0, self._act_eps)]
            self._act_eps = PiecewiseSchedule(self._act_eps)
        
        self._to_sync = Every(self._target_update_period)

    @override(BaseAgent)
    def _construct_optimizers(self):
        raise NotImplementedError
    
    @override(BaseAgent)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to initialize variables
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=((env.action_dim,), tf.float32, 'action'),
            reward=((), tf.float32, 'reward'),
            next_obs=(env.obs_shape, env.obs_dtype, 'next_obs'),
            discount=((), tf.float32, 'discount'),
        )
        if self._is_per:
            TensorSpecs['IS_ratio'] = ((), tf.float32, 'IS_ratio')
        if self._n_steps > 1:
            TensorSpecs['steps'] = ((), tf.float32, 'steps')
        self.learn = build(self._learn, TensorSpecs, batch_size=self._batch_size)
    
    @tf.function
    def _sync_nets(self):
        tvars = self.target_encoder.variables + self.target_q.variables
        mvars = self.encoder.variables + self.q.variables
        [tvar.assign(mvar) for tvar, mvar in zip(tvars, mvars)]

    """ Call """
    @override(BaseAgent)
    def _process_input(self, obs, evaluation, env_ouput):
        obs, kwargs = super()._process_input()
        kwargs['epsilon'] = self._get_eps(evaluation)
        return obs, kwargs

    def _get_eps(self, evaluation):
        if evaluation:
            eps = self._eval_act_eps
        else:
            if self._schedule_act_eps:
                eps = self._act_eps.value(self.env_step)
                self.store(act_eps=eps)
            else:
                eps = self._act_eps
        return eps

    @step_track
    def learn_log(self, step):
        for _ in range(self.N_UPDATES):
            with TBTimer('sample', 2500):
                data = self.dataset.sample()

            if self._is_per:
                idxes = data.pop('idxes').numpy()

            with TBTimer('learn', 2500):
                terms = self.learn(**data)
            if self._to_sync(self.train_step):
                self._sync_target_nets()

            terms = {f'train/{k}': v.numpy() for k, v in terms.items()}
            if self._is_per:
                self.dataset.update_priorities(terms['train/priority'], idxes)

            self.store(**terms)

        if self._to_summary(step):
            self._summary(data, terms)
        
        return self.N_UPDATES

    
    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        priority += self._per_epsilon
        priority **= self._per_alpha
        return priority

    def reset_noisy(self):
        pass