# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.8.10 (default, May 19 2021, 13:12:57) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: player.py
import random
import numpy as np
from .action import Action
from core.elements.agent import Agent
from .utils import get_action_id
from .infoset import get_obs
from env.typing import EnvOutput


class Player(object):

    def __init__(self, name, agent=None):
        self.name = name
        if isinstance(agent, Agent):
            self.agent = agent
        elif agent is None or agent == 'random':
            from env.guandan.random_agent import RandomAgent
            self.agent = RandomAgent()
        elif agent == 'reyn':
            from algo.gd.ruleAI.reyn_ai import ReynAIAgent
            self.agent = ReynAIAgent()
        else:
            raise ValueError(f'Agent({agent}) is not valid')
        self.reset()

    def reset(self):
        self.first_round = True
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
        self.last_pid = -1
        self.last_action = Action()
        self.last_valid_pid = -1
        self.last_valid_action = Action()
        self.is_last_action_first_move = False
        self.action_index = -1
        self.legal_actions = None
        self.history = []

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
        if isinstance(self.agent, Agent):
            obs = get_obs(infoset)
            batch_obs = obs.copy()
            batch_obs['eid'] = 0
            for k, v in batch_obs.items():
                batch_obs[k] = np.expand_dims(v, 0)
            reward = np.expand_dims(0, 0)
            discount = np.expand_dims(1, 0)
            reset = np.expand_dims(self.first_round, 0)
            env_output = EnvOutput(batch_obs, reward, discount, reset)
            action = Agent(env_output)
            action = [np.squeeze(a) for a in action]
            self.action_index = get_action_id(action, infoset)
        else:
            self.action_index = self.agent(infoset)
        return self.action_index

    def tribute_choice(self):
        self.action_index = random.randint(0, len(self.legal_actions)-1)
        return self.action_index

    def back_choice(self):
        self.action_index = random.randint(0, len(self.legal_actions)-1)
        return self.action_index

    @property
    def red_joker_num(self):
        num = 0
        for card in self.hand_cards:
            if card == 'HR':
                num += 1

        return num
# okay decompiling player.pyc
