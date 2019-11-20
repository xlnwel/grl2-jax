import numpy as np
import tensorflow as tf


class Runner():
    def __init__(self, env):
        self.env = env

    def sample_trajectory(self, actor, fn=None):
        """ Sample trajectories
        Args:
            actor: the model responsible for taking actions
            fn: a function that specifies what to do after each env step
        """
        env = self.env
        state = env.reset()

        for _ in range(env.max_episode_steps):
            action = actor.step(tf.convert_to_tensor(state[None], tf.float32))
            next_state, reward, done, _ = env.step(action.numpy())
            if fn:
                fn(state, action, reward, done)
            state = next_state
            if done:
                break
        return env.get_score(), env.get_epslen()

    def random_sampling(self, buffer):
        """ Interact with the environment with random actions to 
        collect data for buffer initialization 
        """
        env = self.env
        state = env.reset()

        while not buffer.good_to_learn:
            for _ in range(env.max_episode_steps):
                action = env.random_action()
                next_state, reward, done, _ = env.step(action)
                buffer.add(state, action, reward, done)
                state = next_state
                if done:
                    break
        