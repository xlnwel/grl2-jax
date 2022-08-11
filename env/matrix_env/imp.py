import numpy as np
from gym.spaces import Discrete

from env.utils import *


class IteratedMatchingPennies:
    """
    A two-agent vectorized environment.
    Possible actions for each agent are (C)ooperate and (D)efect.
    """
    # Possible actions
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 7

    def __init__(self, max_episode_steps, uid2aid, **kwargs):
        self.max_episode_steps = max_episode_steps

        self.uid2aid = uid2aid
        self.aid2uids = compute_aid2uids(self.uid2aid)
        self.n_units = len(self.uid2aid)
        self.n_agents = len(self.aid2uids)

        self.payout_mat1 = np.array([[1., -1.], [-1., 1.]])
        self.payout_mat2 = np.array([[-1., 1.], [1., -1.]])

        self.action_space = [
            Discrete(self.NUM_ACTIONS) for _ in range(self.NUM_AGENTS)
        ]
        self.action_shape = [a.shape for a in self.action_space]
        self.action_dim = [a.n for a in self.action_space]
        self.action_dtype = [np.int32 for _ in self.action_space]
        self.is_action_discrete = [True for _ in self.action_space]

        self.obs_shape = [dict(obs=(self.NUM_STATES,)) for _ in range(self.NUM_AGENTS)]
        self.obs_dtype = [dict(obs=np.float32) for _ in range(self.NUM_AGENTS)]

        self.step_count = None
        self._dense_score = np.zeros(self.NUM_ACTIONS, np.float32)
        self._score = np.zeros(self.NUM_ACTIONS, np.float32)
        self._coop_score = np.zeros(self.NUM_ACTIONS, np.float32)
        self._coop_def_score = np.zeros(self.NUM_ACTIONS, np.float32)
        self._defect_score = np.zeros(self.NUM_ACTIONS, np.float32)
        self._epslen = 0

    def seed(self, seed=None):
        pass

    def random_action(self):
        return [np.random.randint(d) for d in self.action_dim]

    def reset(self):
        self.step_count = 0
        self._dense_score = np.zeros(self.NUM_ACTIONS, np.float32)
        self._score = np.zeros(self.NUM_ACTIONS, np.float32)
        self._coop_score = np.zeros(self.NUM_ACTIONS, np.float32)
        self._coop_def_score = np.zeros(self.NUM_ACTIONS, np.float32)
        self._defect_score = np.zeros(self.NUM_ACTIONS, np.float32)
        self._epslen = 0
        self._defect_step = []

        obs = self._get_obs()

        return obs

    def step(self, action):
        a0, a1 = action
        if isinstance(a0, (list, tuple, np.ndarray)):
            a0 = a0[0]
            a1 = a1[0]

        self.step_count += 1

        obs = self._get_obs([a0, a1])
        
        reward = np.array([self.payout_mat1[a0, a1], self.payout_mat2[a0, a1]], np.float32)
        done = self.step_count == self.max_episode_steps
        dones = np.ones(self.NUM_AGENTS, np.float32) * done

        self._dense_score += reward
        self._coop_score += a0 == a1 == 0
        self._coop_def_score += a0 != a1
        self._defect_score += a0 == a1 == 1
        self._score += reward
        self._epslen += 1

        reward = [reward[u] for u in self.aid2uids]
        dones = [dones[u] for u in self.aid2uids]
        info = {
            'dense_score': self._dense_score, 
            'score': self._score, 
            'epslen': self._epslen, 
            'coop_score': self._coop_score, 
            'coop_defect_score': self._coop_def_score, 
            'defect_score': self._defect_score, 
            'game_over': done, 
        }

        return obs, reward, dones, info

    def _get_obs(self, action=None):
        obs = [{
            'obs': np.zeros((len(u), self.NUM_STATES), np.float32)
        } for u in self.aid2uids]
        if action is None:
            for o in obs:
                o['obs'][:, -1] = 1
        else:
            a0, a1 = action
            if self.n_agents == 1:
                obs[0]['obs'][0, a0 * 2 + a1] = 1
                obs[0]['obs'][1, a1 * 2 + a0] = 1
            else:
                obs[0]['obs'][:, a0 * 2 + a1] = 1
                obs[1]['obs'][:, a1 * 2 + a0] = 1

        if self.n_agents == 1:
            obs[0]['obs'][0, -3] = 1
            obs[0]['obs'][1, -2] = 1
        else:
            obs[0]['obs'][0, -3] = 1
            obs[1]['obs'][0, -2] = 1

        return obs

    def close(self):
        pass