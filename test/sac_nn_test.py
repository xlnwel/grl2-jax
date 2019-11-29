import numpy as np

from algo.sac import nn
from utility.yaml_op import load_config

config = load_config('algo/sac/config.yaml')
config = config['model']
state_shape = (3,)
action_dim = 2
is_action_discrete = False

sac = nn.SAC(config, state_shape, action_dim, is_action_discrete)
actor = sac['actor']

class TestClass:
    def test_get_weights(self):
        assert len(sac.get_weights()) == len(sac.trainable_variables)
        target_vars = np.array([v.numpy() for v in sac.variables])

        for var, tvar in zip(sac.get_weights(), target_vars):
            np.testing.assert_allclose(var, tvar)

    def test_set_weights(self):
        target_sac = nn.SAC(config, state_shape, action_dim, is_action_discrete)
        target_actor = target_sac['actor']
        target_sac.set_weights(sac.get_weights())

        for var, tvar in zip(sac.get_weights(), target_sac.get_weights()):
            np.testing.assert_allclose(var, tvar)

        for var, tvar in zip(actor.get_weights(), target_actor.get_weights()):
            np.testing.assert_allclose(var, tvar)