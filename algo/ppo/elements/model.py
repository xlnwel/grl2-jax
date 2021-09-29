import os
import tensorflow as tf

from core.module import Model
from utility.file import source_file

source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class PPOModel(Model):
    @tf.function
    def action(self, obs, evaluation=False, **kwargs):
        print(f'instantiate action with obs({obs}),',
              f'evaluation({evaluation})',
              f'kwargs({kwargs})')
        x = self.encode(obs)
        act_dist = self.actor(x, evaluation=evaluation)
        action = self.actor.action(act_dist, evaluation)

        if evaluation:
            return action, {}, None
        else:
            logpi = act_dist.log_prob(action)
            if hasattr(self, 'value_encoder'):
                x = self.value_encoder(obs)
            value = self.value(x)
            terms = {'logpi': logpi, 'value': value}

            return action, terms, None    # keep the batch dimension for later use

    @tf.function
    def compute_value(self, x):
        if hasattr(self, 'value_encoder'):
            x = self.value_encoder(x)
        else:
            x = self.encoder(x)
        value = self.value(x)
        return value

    def encode(self, x):
        return self.encoder(x)


def create_model(config, env_stats, name='ppo'):
    config['actor']['action_dim'] = env_stats.action_dim
    config['actor']['is_action_discrete'] = env_stats.action_dim

    return PPOModel(config=config, name=name)
