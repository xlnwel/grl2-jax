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
            action, logpi, value = model(tf.convert_to_tensor(state, tf.float32))
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
            
        _, _, last_value = model(state)
        self.buffer.finish(last_value.numpy())

        score, epslen = env.get_score(), env.get_epslen()

        return self.buffer, score, epslen

    def evaluate(self, train_epoch):
        """
        Evaluate the learned policy in another process. 
        This function should only be called at the training time.
        """
        i = 0
        while i < 100:
            i += env.n_envs
            state = env.reset()
            for j in range(env.max_episode_steps):
                action = self.ac.det_action(tf.convert_to_tensor(state, tf.float32))
                state, _, done, _ = env.step(action.numpy())

                if np.all(done):
                    break
            self.store(score=env.get_score(), epslen=env.get_epslen())

        stats = dict(
                model_name=f'{self.model_name}',
                timing='Eval',
                steps=f'{train_epoch}',
        )
        stats.update(self.get_stats(mean=True, std=True))
        self.log_stats(stats)