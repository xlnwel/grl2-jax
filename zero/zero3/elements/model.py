import os
import tensorflow as tf

from core.tf_config import build
from tools.file import source_file
from algo.zero.elements.model import ModelImpl, MAPPOModelEnsemble

source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class MAPPOActorModel(ModelImpl):
    def action(self, obs, goal, state=None, mask=None, action_mask=None,
            evaluation=False, return_eval_stats=False):
        x, state = self.encode(obs, state, mask)
        goal = self.goal_encoder(goal)
        act_dist = self.policy(x, goal, action_mask, evaluation=evaluation)
        action = self.policy.action(act_dist, evaluation)

        if evaluation:
            # we do not compute the value state at evaluation 
            return action, {}, state
        else:
            logpi = act_dist.log_prob(action)
            terms = {'logpi': logpi}
            return action, terms, state


class MAPPOValueModel(ModelImpl):
    def compute_value(self, global_state, state=None, mask=None):
        x, state = self.encode(global_state, state, mask)
        value = self.value(x)
        return value, state


class MAPPOWithGoalModelEnsemble(MAPPOModelEnsemble):
    def _build(self, env_stats, evaluation=False):
        basic_shape = (None, len(env_stats.aid2uids[self.config.aid]))
        dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        actor_inp=dict(
            obs=((*basic_shape, *env_stats['obs_shape'][self.config.aid]['obs']), 
                env_stats['obs_dtype'][self.config.aid]['obs'], 'obs'),
            goal=((*basic_shape, *env_stats['obs_shape'][self.config.aid]['goal']), 
                env_stats['obs_dtype'][self.config.aid]['goal'], 'goal'),
        )
        value_inp=dict(
            global_state=(
                (*basic_shape, *env_stats['obs_shape'][self.config.aid]['global_state']), 
                env_stats['obs_dtype'][self.config.aid]['global_state'], 'global_state'),
        )
        TensorSpecs = dict(
            actor_inp=actor_inp,
            value_inp=value_inp,
            evaluation=evaluation,
            return_eval_stats=evaluation,
        )
        if self.has_rnn:
            actor_inp['mask'] = (basic_shape, tf.float32, 'mask')
            value_inp['mask'] = (basic_shape, tf.float32, 'mask')
            TensorSpecs.update(dict(
                actor_state=self.actor_state_type(*[((None, sz), dtype, name) 
                    for name, sz in self.actor_state_size._asdict().items()]),
                value_state=self.value_state_type(*[((None, sz), dtype, name) 
                    for name, sz in self.value_state_size._asdict().items()]),            
            ))
        if env_stats.use_action_mask:
            TensorSpecs['actor_inp']['action_mask'] = (
                (*basic_shape, env_stats.action_dim), tf.bool, 'action_mask'
            )
        self.action = build(self.action, TensorSpecs)


def create_model(
        config, 
        env_stats, 
        name='mappo', 
        to_build=False,
        to_build_for_eval=False,
        **kwargs):
    aid = config['aid']
    config.policy.policy.action_dim = env_stats.action_dim[aid]
    config.policy.policy.is_action_discrete = env_stats.is_action_discrete[aid]

    return MAPPOWithGoalModelEnsemble(
        config=config, 
        env_stats=env_stats, 
        name=name,
        to_build=to_build, 
        to_build_for_eval=to_build_for_eval,
        policy=MAPPOActorModel,
        value=MAPPOValueModel,
        **kwargs
    )
