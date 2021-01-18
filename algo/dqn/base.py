import itertools
import logging
import tensorflow as tf

from utility.utils import Every
from utility.schedule import PiecewiseSchedule
from utility.schedule import TFPiecewiseSchedule
from core.optimizer import Optimizer
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import override, step_track


logger = logging.getLogger(__name__)

def get_data_format(*, env, replay_config, **kwargs):
    is_per = replay_config['replay_type'].endswith('per')
    n_steps = replay_config['n_steps']
    obs_dtype = tf.uint8 if len(env.obs_shape) == 3 else tf.float32
    action_dtype = tf.int32 if env.is_action_discrete else tf.float32
    data_format = dict(
        obs=((None, *env.obs_shape), obs_dtype),
        action=((None, *env.action_shape), action_dtype),
        reward=((None, ), tf.float32), 
        next_obs=((None, *env.obs_shape), obs_dtype),
        discount=((None, ), tf.float32),
    )
    if is_per:
        data_format['IS_ratio'] = ((None, ), tf.float32)
        data_format['idxes'] = ((None, ), tf.int32)
    if n_steps > 1:
        data_format['steps'] = ((None, ), tf.float32)

    return data_format

def collect(replay, env, env_step, reset, **kwargs):
    # if reset:
    #     # we reset noisy every episode. Theoretically, 
    #     # this follows the guide of deep exploration.
    #     # More importantly, it saves time!
    #     if hasattr(agent, 'reset_noisy'):
    #         agent.reset_noisy()
    replay.add(**kwargs)


class DQNBase(BaseAgent):
    """ Initialization """
    @override(BaseAgent)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)

        self._is_per = False if dataset is None else dataset.name().endswith('per')
        self._double = getattr(self, '_double', False)
        self._return_stats = getattr(self, '_return_stats', False)
        self._schedule_act_eps = env.n_envs == 1 \
            and getattr(self, '_schedule_act_eps', False)

        if self._schedule_act_eps:
            assert isinstance(self._act_eps, (list, tuple)), self._act_eps
            logger.info(f'Schedule action epsilon: {self._act_eps}')
            self._act_eps = PiecewiseSchedule(self._act_eps)
        
        if hasattr(self, '_target_update_period'):
            self._to_sync = Every(self._target_update_period)
    
    @override(BaseAgent)
    def _construct_optimizers(self):
        if self._schedule_lr:
            assert isinstance(self._lr, list), self._lr
            self._lr = TFPiecewiseSchedule(self._lr)
        models = [v for k, v in self.model.items() if 'target' not in k]
        self._optimizer = Optimizer(
            self._optimizer, models, self._lr, 
            weight_decay=getattr(self, '_weight_decay', None),
            clip_norm=getattr(self, '_clip_norm', None),
            epsilon=getattr(self, '_epsilon', 1e-7))

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
        ons = self.get_online_nets()
        tns = self.get_target_nets()
        logger.info(f"Online networks: {[n.name for n in ons]}")
        logger.info(f"Target networks: {[n.name for n in tns]}")
        ovars = list(itertools.chain(*[v.variables for v in ons]))
        tvars = list(itertools.chain(*[v.variables for v in tns]))
        [tvar.assign(ovar) for tvar, ovar in zip(tvars, ovars)]
    
    def get_online_nets(self):
        return [getattr(self, f'{k}') for k in self.model 
            if f'target_{k}' in self.model]

    def get_target_nets(self):
        return [getattr(self, f'target_{k}') for k in self.model 
            if f'target_{k}' in self.model]

    """ Call """
    @override(BaseAgent)
    def _process_input(self, obs, evaluation, env_output):
        obs, kwargs = super()._process_input(obs, evaluation, env_output)
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
        return tf.convert_to_tensor(eps, tf.float32)

    @step_track
    def learn_log(self, step):
        for _ in range(self.N_UPDATES):
            with self._sample_timer:
                data = self.dataset.sample()

            if self._is_per:
                idxes = data.pop('idxes').numpy()

            with self._train_timer:
                terms = self.learn(**data)

            if self._to_sync(self.train_step):
                self._sync_nets()

            terms = {f'train/{k}': v.numpy() for k, v in terms.items()}
            if self._is_per:
                self.dataset.update_priorities(terms['train/priority'], idxes)

            self.store(**terms)

        if self._to_summary(step):
            self._summary(data, terms)
        
        self.store(**{
            'time/sample': self._sample_timer.average(),
            'time/train': self._train_timer.average()
        })

        return self.N_UPDATES
    
    def _compute_priority(self, priority):
        """ p = (p + 𝝐)**𝛼 """
        priority += self._per_epsilon
        priority **= self._per_alpha
        return priority

    def reset_noisy(self):
        pass