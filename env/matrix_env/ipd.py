import numpy as np

from gym.spaces import Discrete
from env.utils import *

class IteratedPrisonersDilemma:
    """
    A two-agent vectorized environment.
    Possible actions for each agent are (C)ooperate and (D)efect.
    """
    # Possible actions
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 5

    def __init__(self, max_episode_steps, uid2aid, **kwargs):
        self.max_episode_steps = max_episode_steps
        self.payout_mat = np.array([[-1., 0.], [-3., -2.]])

        self.uid2aid = uid2aid
        self.aid2uids = compute_aid2uids(self.uid2aid)
        self.n_units = len(self.uid2aid)
        self.n_agents = len(self.aid2uids)

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

        obs = {'obs': np.zeros((self.NUM_AGENTS, self.NUM_STATES), np.float32)}
        obs['obs'][:, -1] = 1

        return [obs]

    def step(self, action):
        a0, a1 = action

        self.step_count += 1

        obs = {'obs': np.zeros((self.NUM_AGENTS, self.NUM_STATES), np.float32)}
        
        reward = np.array([self.payout_mat[a1][a0], self.payout_mat[a0][a1]], np.float32)
        obs['obs'][0, a0 * 2 + a1] = 1
        obs['obs'][1, a1 * 2 + a0] = 1
        
        self._dense_score += reward
        self._coop_score += a0 == a1 == 0
        self._coop_def_score += a0 != a1
        self._defect_score += a0 == a1 == 1
        self._score += reward
        self._epslen += 1
        if self._epslen == 76:
            print('action at step 76', a0, a1)

        done = self.step_count == self.max_episode_steps
        dones = np.ones(self.NUM_AGENTS, np.float32) * done
        info = {
            'dense_score': self._dense_score, 
            'score': self._score, 
            'epslen': self._epslen, 
            'coop_score': self._coop_score, 
            'coop_defect_score': self._coop_def_score, 
            'defect_score': self._defect_score, 
            'game_over': done, 
            'defect_step': self._defect_step,
        }

        return [obs], [reward], [dones], info
