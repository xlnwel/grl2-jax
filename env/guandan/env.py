import numpy as np

from env.guandan.player import Player
from env.guandan.game import Game
from env.guandan.infoset import get_obs


class Env:
    def __init__(self):
        # Initialize players
        # We use for dummy player for the target position
        # Initialize the internal environment
        players = [Player('0'), Player('1'), Player('2'), Player('3')]
        self._env = Game(players)
        self.max_episode_steps = int(1e9)
        self.is_action_discrete = True
        self.obs_shape = self._env.obs_shape
        self.action_shape = self._env.action_shape
        self.action_dim = self._env.action_dim
        self.obs_dtype = self._env.obs_dtype
        self.action_dtype = self._env.action_dtype

    def random_action(self):
        return self._env.random_action()

    def reset(self):
        self._env.reset()
        self._env.start()

        return get_obs(self._get_infoset())

    def step(self, action):
        #self._env.players[self._acting_player_position].set_action(action)
        self._env.play(action)
        done = False
        reward = 0.0
        info = {}

        if self.game_over():
            done = True
            reward = self._env.compute_reward()
            obs = None
        else:
            obs = get_obs(self._get_infoset)

        return obs, reward, done, info

    def _get_infoset(self):
        """
        Here, infoset is defined as all the information
        in the current situation, including the hand cards
        of all the players, all the historical moves, etc.
        That is, it contains perfect information.
        """
        return self._env.get_infoset()

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