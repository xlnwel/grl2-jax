import numpy as np

from algo.sac import nn
from utility.yaml_op import load_config

config = load_config('algo/sac/config.yaml')
config = config['model']
state_shape = (3,)
action_dim = 2
is_action_discrete = False

actor = nn.SoftPolicy(config['actor'], state_shape, action_dim, is_action_discrete)


class TestClass:
    def test_get_weights(self):
        assert len(actor.get_weights()) == len(actor.trainable_variables)
        target_vars = np.array([v.numpy() for v in actor.variables])

        for var, tvar in zip(actor.get_weights(), target_vars):
            np.testing.assert_allclose(var, tvar)

    def test_set_weights(self):
        target_actor = nn.SoftPolicy(config['actor'], state_shape, action_dim, is_action_discrete)

        target_actor.set_weights(actor.get_weights())

        for var, tvar in zip(actor.get_weights(), target_actor.get_weights()):
            np.testing.assert_allclose(var, tvar)
