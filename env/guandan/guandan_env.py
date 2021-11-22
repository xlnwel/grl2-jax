import numpy as np
from env.guandan.player import Player
from env.guandan.small_game import SmallGame
from env.guandan.action import Action
from env.guandan.infoset import get_obs

class Env:
    def __init__(self):
        """
        """
        # Initialize players
        # We use for dummy player for the target position
        # Initialize the internal environment
        #self.players = [Player('0'), Player('1'), Player('2'), Player('3')]
        players = [Player('0'), Player('1'), Player('2'), Player('3')]
        self._env = SmallGame()
        self._env.players = players

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
