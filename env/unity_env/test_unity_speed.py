import time
from copy import deepcopy

import numpy as np
# from interface import UnityInterface
from mlagents_envs.environment import UnityEnvironment
# from mlagents_envs.base_env import ActionTuple

n_copy = 1


# def step(env, action):
#     name = ['Player']
#
#     action_tuple = env.get_action_tuple()
#
#     for i in range(n_copy):
#         for id, n in enumerate(name):
#             if id in [1, 2, 3, 4]:
#                 action_tuple.add_discrete(np.zeros((1, 2)))
#             else:
#                 action_tuple.add_discrete(np.zeros((1, 4)))
#
#             action_tuple.add_continuous(np.hstack((action.reshape((1, 5)), np.zeros((1, 2)))))
#             env.set_actions(f'Red?team=0', deepcopy(action_tuple))
#     # self.env.set_actions('E0_Blue_0?team=0', action_tuple)
#     return env.step()

#
# def step(env):
#     action_tuple = ActionTuple()
#     # action_tuple.add_discrete(np.random.randint(0, 3, size=(1, 3)))
#     # env.set_action_for_agent(f'SoccerTwos?team=1', 2, deepcopy(action_tuple))
#     #action_tuple.add_discrete(np.random.randint(0, 3, size=(1, 3)))
#     action_tuple.add_discrete(np.ones((1, 3)) * 2)
#     env.set_action_for_agent(f'SoccerTwos?team=1', 2, deepcopy(action_tuple))
#     # action_tuple.add_discrete(np.random.randint(0, 3, size=(1, 3)))
#     # env.set_action_for_agent(f'SoccerTwos?team=0', 1, deepcopy(action_tuple))
#     # action_tuple.add_discrete(np.random.randint(0, 3, size=(1, 3)))
#     # env.set_action_for_agent(f'SoccerTwos?team=0', 3, deepcopy(action_tuple))
#
#     env.step()
#     return env.get_steps('SoccerTwos?team=1')


def reset(env):
    env.reset()


env = UnityEnvironment()

env.reset()
start_time = time.time()
for i in range(1, 100):
    print(i)
    # env.step()
    # ds = step(env)
    ds = env.step(np.array([0, 0, 0, 0.5, 1]))
    end_time = time.time()
print(end_time - start_time)
env.reset()
