# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.8.10 (default, May 19 2021, 13:12:57) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: player.py
import json
from guandan.state import State
import random

class Player(object):

    def __init__(self, name, action=None):
        self.name = name
        self.index = None
        self.victory = 0
        self.draws = 0
        self.uni_count = 0
        self.hand_cards = []
        self.play_area = ''
        self.rank = 2
        self.stuck_times = 0
        self.reward = 0
        self.public_info = None
        self.oppo_rank = 2
        self.greater_pos = -1
        self.greater_action = None
        self.action_index = -1
        self.action_list = None
        self.history = []
        if action is None:
            from guandan.random_agent import RandomAgent
            self.agent = RandomAgent()
        if action =='random':
            from guandan.random_agent import RandomAgent
            self.agent = RandomAgent()
        if action =='deep':
            from dmc.deep_agent import DeepAgent
            self.agent = DeepAgent()
        if action == 'rule':
            from rule_agent import RuleAgent
            self.agent = RuleAgent()

    def play(self, cards, rank):
        try:
            for card in cards:
                self.hand_cards.remove(card)
                if card == 'H{}'.format(rank):
                    self.uni_count -= 1
        except ValueError:
            print(cards)
            print(self.hand_cards)
            print(self.uni_count)
            raise ValueError

    def get_public_info(self):
        return {'rest':len(self.hand_cards),  'playArea':self.play_area, 'history':self.history}

    def update_rank(self, inc):
        self.rank += inc
        self.reward += inc
        self.rank = min(self.rank, 14)

    def clear_play_area(self):
        self.play_area = None

    def play_choice(self, infoset):
        self.action_index = self.agent.act(infoset)
        return self.action_index

    def tribute_choice(self):
        self.action_index = random.randint(0, len(self.action_list)-1)
        return self.action_index

    def back_choice(self):
        self.action_index = random.randint(0, len(self.action_list)-1)
        return self.action_index

    @property
    def red_joker_num(self):
        num = 0
        for card in self.hand_cards:
            if card == 'HR':
                num += 1

        return num
# okay decompiling player.pyc
