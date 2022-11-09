import numpy as np
import matplotlib.pyplot as plt

import gym

env_name = "Pendulum-v1"
env = gym.make(env_name)
print(env)
# print("init env state: ", env.state)

obs = env.reset()
print("init env state: ", env.state)


theta, thetadot = np.pi / 2, 0.
# env.set_state(theta, thetadot)
env.state = np.array([theta, thetadot])
print("env.state: ", env.state)
print("env.env.state: ", env.env.state)
# env.print_addr()


# img = env.render(mode='rgb_array')
# # plt.imshow(img)
# plt.savefig('pendulum.png', bbox_inches='tight')
