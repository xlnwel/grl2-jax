from collections import deque
import numpy as np

from .game import Game
from .infoset import get_obs
from .utils import get_action_id


class Env:
    def __init__(self, eid, **kwargs):
        # Initialize players
        # We use for dummy player for the target position
        # Initialize the internal environment
        self.eid = eid
        self._env = Game(**kwargs)
        self.evaluation = kwargs.get('evaluation', False)
        self.n_players = 4
        self.max_episode_steps = int(1e9)
        self.is_action_discrete = True
        self.obs_shape = self._env.obs_shape
        self.action_shape = self._env.action_shape
        self.action_dim = self._env.action_dim
        self.reward_shape = self._env.reward_shape
        self.obs_dtype = self._env.obs_dtype
        self.action_dtype = self._env.action_dtype
        self.players_states = deque([np.zeros(128, dtype=np.float32) for _ in range(3)], 3)

    def random_action(self):
        return self._env.random_action()

    def reset(self):
        self._env.reset()
        self._env.start()

        obs = get_obs(self._get_infoset())
        obs['eid'] = self.eid
        if not self.evaluation:
            obs['others_h'] = self._get_others_state()

        return obs

    def step(self, action):
        if not self.evaluation:
            action, state = action[:2], action[2]
            self.players_states.append(state)
        aid = self._get_action_id(action)
        self._env.play(aid)
        done = False
        reward = np.zeros(4, dtype=np.float32)
        info = {}

        if self.game_over():
            done = True
            reward = self._env.compute_reward()
            info['won'] = reward[0] > 0
            info['score'] = reward[0]
            info['game_over'] = True
        obs = get_obs(self._get_infoset())
        obs['eid'] = self.eid
        if not self.evaluation:
            obs['others_h'] = self._get_others_state()

        return obs, reward, done, info

    def _get_infoset(self):
        """
        Here, infoset is defined as all the information
        in the current situation, including the hand cards
        of all the players, all the historical moves, etc.
        That is, it contains perfect information.
        """
        self.infoset = self._env.get_infoset()
        return self.infoset

    def _get_others_state(self):
        h = np.stack(self.players_states)
        return h

    @property
    def _acting_player_position(self):
        """
        The player that is active.
        """
        return self._env.current_pos

    def game_over(self):
        """ Returns a Boolean
        """
        return self._env.game_over()

    def close(self):
        self._env.close()
    
    def _get_action_id(self, action):
        return get_action_id(action, self.infoset)

    def get_action_id(self, action_type, card_rank):
        return self.infoset.action2id(action_type, card_rank)
