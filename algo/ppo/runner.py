import numpy as np
import tensorflow as tf


class Runner():
    def __init__(self, env, buffer):
        self.env = env
        self.buffer = buffer

    def sample_trajectories(self, model):
        self.buffer.reset()
        model.reset_states()
        env = self.env
        state = env.reset()

        for _ in range(self.env.max_episode_steps):
            action, logpi, value = model.step(tf.convert_to_tensor(state, tf.float32))
            next_state, reward, done, _ = env.step(action.numpy())
            self.buffer.add(state=state, 
                            action=action.numpy(), 
                            reward=reward, 
                            value=value.numpy(), 
                            old_logpi=logpi.numpy(), 
                            nonterminal=1-done, 
                            mask=env.get_mask())
            
            state = next_state
            if np.all(done):
                break
            
        _, _, last_value = model.step(tf.convert_to_tensor(state, tf.float32))
        self.buffer.finish(last_value.numpy())

        score, epslen = env.get_score(), env.get_epslen()

        return self.buffer, score, epslen
