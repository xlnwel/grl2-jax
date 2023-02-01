from .gym_multigrid import REGISTRY as MINIGRID_REGISTRY
import gym
import numpy as np


class MAMiniGrid(gym.Wrapper):
    def __init__(self, config):
        self.env = MINIGRID_REGISTRY[config.env_name](
            **config.env_args
        )

        self.n_agents = self.env.num_agents
        self.uid2aid = list(range(self.n_agents))
        self.n_units = self.n_agents

        self.observation_space = [self.env.observation_space for _ in range(self.n_agents)]
        self.action_space = [self.env.action_space for _ in range(self.n_agents)]
        self.action_shape = [a.shape for a in self.action_space]
        self.action_dim = [a.n for a in self.action_space]
        self.action_dtype = [np.int32 for _ in self.action_space]
        self.is_action_discrete = True

        self.obs_shape = [{
            'obs': (self.env.ob_dim, ),
            'global_state': (self.env.state_dim, ),
        } for _ in range(self.n_agents)]
        self.obs_dtype = [{
            'obs': np.float32,
            'global_state': np.float32,
        } for _ in range(self.n_agents)]
        
        self.max_episode_steps = config.max_episode_steps 
        self._score = np.zeros(self.n_agents)
        self._dense_score = np.zeros(self.n_agents)
        self._epslen = 0

    def step(self, actions):
        actions = np.reshape(actions, (self.n_units, -1))
        # done is scalar.
        obs, reward, done, _ = self.env.step(actions)
        state = self.env.get_state()

        self._score += reward[0]
        self._dense_score += reward[0]
        self._epslen += 1

        info = {
            'score': self._score,
            'dense_score': self._dense_score,
            'epslen': self._epslen,
            'game_over': self._epslen == self.max_episode_steps
        }

        done = np.array([done for _ in range(self.n_agents)])

        obs = {
            'obs': np.stack(obs),
            'global_state': np.stack([state for _ in range(self.n_units)]),
        }

        assert len(obs) == self.n_agents, (obs, self.n_agents)
        assert len(reward) == self.n_agents, (reward, self.n_agents)
        assert len(done) == self.n_agents, (done, self.n_agents)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        assert len(obs) == self.n_agents, (obs, self.n_agents)

        self._score = np.zeros(self.n_agents)
        self._dense_score = np.zeros(self.n_agents)
        self._epslen = 0

        state = self.env.get_state()
        
        obs = {
            'obs': np.stack(obs),
            'global_state': np.stack([state for _ in range(self.n_units)]),
        }
        
        return obs