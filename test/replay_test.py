import numpy as np

from env.gym_env import create_gym_env
from replay.func import create_replay


n_steps = 3
gamma = .99
capacity = 15
config = dict(
    type='proportional',
    beta0=.4,
    beta_steps=5e4,
    n_steps=n_steps,
    gamma=gamma,
    batch_size=3,
    min_size=7,
    capacity=capacity,
)

keys = ['state', 'action', 'reward', 'done', 'n_steps']
state_shape(4, )
class TestClass:
    def test_buffer_op(self):
        replay = create_replay(config, *keys, state_shape)

        