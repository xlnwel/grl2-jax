import numpy as np

from algo.ppo.agent import Agent as PPOAgent

class Agent(PPOAgent):
    def train_record(self, step):
        reward = self.dataset._memory['reward']
        kl_reward = self.dataset._kl_coef * self.dataset._memory['kl']
        self.store(**{
            'reward': np.mean(reward), 
            'kl_reward': np.mean(kl_reward)})

        train_steps = super().train_record(step)
        return train_steps